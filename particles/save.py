import logging
from math import floor

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from descartes import PolygonPatch
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def save_animation(file_name, bbox, continent, current_df, states, num_iter, step):
    # создаем рисунок
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('x, м')
    ax.set_ylabel('y, м')

    ax.set_xlim([bbox[0], bbox[2]])
    ax.set_ylim([bbox[1], bbox[3]])
    ax.add_patch(PolygonPatch(continent))
    ax.plot(*continent.exterior.xy)
    ax.axis('equal')

    # добавляем поле скоростей
    df = current_df[(current_df['time'] == 1) & (current_df['depth'] == 1)]
    q = ax.quiver(
        df.loc[:, 'x'],
        df.loc[:, 'y'],
        df.loc[:, 'uo'],
        df.loc[:, 'vo'],
        scale=10,
        scale_units='width',
        cmap='winter',
    )

    particles = {}
    transform_offset = transforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.10, units='inches')
    states_df = pd.read_csv('out.csv')
    for particle_id, group_df in states_df.groupby('id'):
        logger.debug('Рисуем частицу %s', particle_id)
        particles[particle_id] = {'line': ax.plot([], [], linestyle='-', marker='', color='red')[0],
                                  'point': ax.plot([], [], linestyle='', marker='o')[0],
                                  'text': ax.text(0, 0, '', transform=transform_offset)}

    def init():
        """Initialize lines"""
        result = []
        for line in particles.values():
            line['line'].set_data([0, 0], [0, 0])
            line['point'].set_data([0, 0], [0, 0])
            line['text'].set_text('')
            line['text'].set_x(0)
            line['text'].set_y(0)
            result.append(line['line'])
            result.append(line['point'])
            result.append(line['text'])
        return result

    def update(frame):
        """Update frame"""
        logger.debug('Рисуем кадр %s', frame)
        ax.set_title('{}, час'.format(frame * step // 3600))

        day = 1 + floor(frame * step / 86400)
        df = current_df[(current_df['time'] == 1) & (current_df['depth'] == 1)]
        q.set_UVC(df['uo'], df['vo'])

        result = []
        for particle_id, particle_df in states_df.groupby('id'):
            x = particle_df['x'].values[:frame + 1]
            y = particle_df['y'].values[:frame + 1]
            z = particle_df['z'].values[:frame + 1]
            particles[particle_id]['line'].set_data(x, y)
            particles[particle_id]['point'].set_data(x[-1], y[-1])
            particles[particle_id]['text'].set_text(z[-1])
            particles[particle_id]['text'].set_x(x[-1])
            particles[particle_id]['text'].set_y(y[-1])
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
    logger.debug('Создан файл %s' % file_name)


class CSV:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_obj = open(file_name, 'w')
        self.file_obj.write('id,x,y,z,u,v,w\n')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file_obj.close()
        logger.debug('Создан файл %s' % self.file_name)

    def write(self, particle):
        self.file_obj.write(f'{particle.id},' + ','.join([str(s) for s in particle.state]) + '\n')
