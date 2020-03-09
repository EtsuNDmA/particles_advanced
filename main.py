import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Proj

from particles.continent import Continent
from particles.environment import get_environment_func
from particles.save import CSV, save_animation
from particles.state import Particle, get_current_data, get_particles, init_state
from particles.xlsx_to_csv import xlsx_to_csv

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Зададим константы
NUM_ITER = 24 * 4
STEP = 1800  # с
CONTINENT_BUFFER = 4000  # м
SEED = 42
RATIO = 4000 / 90
SCALE = 4000  # м

Uw, Vw = 100, 100  # см/с
W = np.array([Uw, Vw]).reshape(2, 1)

# Системы координат
WGS84_CRS = Proj(init="EPSG:4326")
LOCAL_CRS = Proj(init="EPSG:2463")


def main(options):
    logger.info(
        f'\n'
        f'#############################\n'
        f'Параметры:\n'
        f'  Число итераций:\t{options.num_iter}\n'
        f'  Шаг итерации, сек:\t{options.step}\n'
        f'  Ширина прибрежной зоны, м:\t{options.continent_buffer}\n'
        f'#############################'
    )
    logger.info('Готовим данные для модели')

    random_state = np.random.RandomState(options.seed)

    if options.force_transform or not Path('data.csv').is_file():
        logger.info('Конвертируем %s в %s', options.input_xlsx, options.input_csv)
        xlsx_to_csv(options.input_xlsx, options.input_csv)
    logger.info('Загружаем течения')
    current_df = get_current_data(options.input_csv)

    # Конвертируем в метры

    min_lat, max_lat = current_df['lat'].min(), current_df['lat'].max()
    min_lon, max_lon = current_df['lon'].min(), current_df['lon'].max()
    max_depth = current_df['depth'].max()

    current_df['x'] = (current_df['lat'] - min_lat) * SCALE
    current_df['y'] = (current_df['lon'] - min_lon) * SCALE

    # Получим границы данных
    bounds = current_df['x'].min(), current_df['y'].min(), current_df['x'].max(), current_df['y'].max()

    logger.info('Создаем континент')
    continent = Continent(current_df, SCALE, options.continent_buffer)

    logger.info('Создаем частицы')
    particles_df = get_particles(options.particles_csv)
    particles = [Particle(p.density, p.diameter,
                          init_state(bounds, random_state),
                          bbox=bounds,
                          max_depth=max_depth,
                          continent=continent,
                          step=options.step)
                 for p in particles_df.itertuples()]
    logger.info(f'Создали {len(particles)} частиц')

    if not options.only_animation:
        logger.info('Запускаем моделирование')
        run_model(options, particles, current_df)

    states = pd.read_csv(options.out_csv)
    logger.info('Сохраняем анимацию')
    save_animation(file_name=options.out_mp4,
                   bbox=bounds,
                   continent=continent.continent,
                   current_df=current_df,
                   states=states[['x', 'y', 'z', 'u', 'v', 'w']],
                   num_iter=options.num_iter,
                   step=options.step)


def run_model(options, particles, current_df):
    with CSV(options.out_csv) as csv_file:
        for iteration in range(options.num_iter):
            logger.info(f'Итерация {iteration}')
            for particle in particles:
                if not particle.is_active:
                    continue
                environment_func = get_environment_func(iteration * options.step, current_df)
                particle.next_state(environment_func)
                csv_file.write(iteration, particle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create animation of particles')
    parser.add_argument('-f', '--force-transform', action='store_true',
                        help='Принудительно конвертирует xls в csv')
    parser.add_argument('-a', '--only-animation', action='store_true',
                        help='Сделать анимацию по данным из out-csv, '
                             'удобно для тестирования настроек анимации без пересчета всей модели')
    parser.add_argument('-n', '--num-iter', type=int, default=NUM_ITER, help='Число итераций')
    parser.add_argument('-s', '--step', type=int, default=STEP, help='Шаг итерации, сек')
    parser.add_argument('-b', '--continent_buffer', type=int, default=CONTINENT_BUFFER,
                        help='Ширина прибрежной зоны, м')
    parser.add_argument('--input-xlsx', type=str, default='data.xlsx', help='Файл с данными xlsx')
    parser.add_argument('--input-csv', type=str, default='data.csv', help='Файл с данными csv')
    parser.add_argument('--particles-csv', type=str, default='particles.csv', help='Файл с частицами csv')
    parser.add_argument('--out-mp4', type=str, default='out.mp4', help='Видео')
    parser.add_argument('--out-csv', type=str, default='out.csv', help='Результаты в csv')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    args = parser.parse_args()

    main(args)
