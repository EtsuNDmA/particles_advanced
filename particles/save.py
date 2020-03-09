import logging
from math import floor

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from descartes import PolygonPatch
from matplotlib import transforms, colors

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_animation(file_name, bbox, continent, current_df, states, num_iter, step):
    # создаем рисунок
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlabel('x, м')
    ax.set_ylabel('y, м')

    ax.set_xlim([bbox[0], bbox[2]])
    ax.set_ylim([bbox[1], bbox[3]])
    ax.add_patch(PolygonPatch(continent))
    ax.axis('equal')

    # добавляем поле скоростей
    df = current_df[(current_df['time'] == 1) & (current_df['depth'] == 1)]
    current_velocity = np.sqrt(df.loc[:, 'uo'] ** 2 + df.loc[:, 'vo'] ** 2)
    q = ax.quiver(
        df.loc[:, 'x'],
        df.loc[:, 'y'],
        df.loc[:, 'uo'],
        df.loc[:, 'vo'],
        current_velocity,
        norm=colors.Normalize(vmin=current_velocity.min(), vmax=current_velocity.max()),
        scale=7,
        scale_units='width',
        cmap='winter',
    )

    particles = {}
    transform_offset = transforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.10, units='inches')
    states_df = pd.read_csv('out.csv')
    for particle_id, group_df in states_df.groupby('id'):
        logger.info('Рисуем частицу %s', particle_id)
        particles[particle_id] = {'line': ax.plot([], [], linestyle='-', marker='', color='red')[0],
                                  'point': ax.plot([], [], linestyle='', marker='o')[0],
                                  'text': ax.text(0, 0, '', transform=transform_offset)}

    def init():
        """Initialize lines"""
        result = [q]
        for line in particles.values():
            result.append(line['line'])
            result.append(line['point'])
            result.append(line['text'])
        return result

    last_day = 0

    def update(frame):
        """Update frame"""
        nonlocal last_day
        logger.info('Рисуем кадр %s', frame)
        ax.set_title('{}, час'.format(frame * step // 3600))

        day = 1 + floor(frame * step / 86400)
        if day > last_day:
            df = current_df[(current_df['time'] == day) & (current_df['depth'] == 1)]
            q.set_UVC(df['uo'], df['vo'], np.sqrt(df['uo'] ** 2 + df['vo'] ** 2))
            last_day = day

        result = [q]
        for particle_id, particle_df in states_df.groupby('id'):
            if frame < particle_df.shape[0]:
                x = particle_df['x'].iloc[:frame + 1]
                y = particle_df['y'].iloc[:frame + 1]
                z = particle_df['z'].iloc[:frame + 1]

                particles[particle_id]['line'].set_data(x, y)
                particles[particle_id]['point'].set_data(x.iat[frame], y.iat[frame])
                particles[particle_id]['point'].set_color(
                    'red' if particle_df['is_active'].iat[frame] else 'black'
                )
                particles[particle_id]['text'].set_text(f'{z.iat[frame]:.2f}')
                particles[particle_id]['text'].set_x(x.iat[frame])
                particles[particle_id]['text'].set_y(y.iat[frame])
            result.append(particles[particle_id]['line'])
            result.append(particles[particle_id]['point'])
            result.append(particles[particle_id]['text'])
        result.append(q)

        return result

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_iter - 1,
        init_func=init,
        interval=100,
        blit=True,
    )
    ani.save(file_name, writer=animation.FFMpegFileWriter())
    logger.info('Создан файл %s' % file_name)


class CSV:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_obj = open(file_name, 'w')
        self.file_obj.write('it,id,is_active,x,y,z,u,v,w\n')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file_obj.close()
        logger.info('Создан файл %s' % self.file_name)

    def write(self, iteration, particle):
        row_data = [str(iteration), str(particle.id), str(particle.is_active)] + [str(s) for s in particle.state]
        self.file_obj.write(','.join(row_data) + '\n')
