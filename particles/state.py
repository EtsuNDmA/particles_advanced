import logging
from itertools import count

import numpy as np
import pandas as pd
import seawater

logger = logging.getLogger(__name__)


g = 978  # см/с2
NU = 17890  # см2/с
RO_W = 1.005  # г/см3
RO_F = 1.388  # г/см3
RO_0 = 0.967  # г/см3
R_0 = 0.25  # см


counter = count()


def get_current_data(file_name):
    """Get current vectors from file"""
    df = pd.read_csv(file_name)

    # replace invalid values with nan
    df[df.loc[:, ['uo', 'vo', 'so', 'thetao']] == 65535] = np.nan

    df['dens'] = seawater.eos80.dens0(df['so'], df['thetao'])
    return df


def get_particles(file_name):
    """Get particles from file"""
    df = pd.read_csv(file_name)
    return df


def init_state(bounds, random_state):
    """Initialize state of the system"""
    x_min, y_min, x_max, y_max = bounds
    x_delta = x_max - x_min
    xmin_, xmax_ = x_min + 0.1 * x_delta, x_min + 0.9 * x_delta
    y_delta = y_max - y_min
    ymin_, ymax_ = y_min + 0.1 * y_delta, y_min + 0.9 * y_delta

    x = (xmax_ - xmin_) * random_state.random_sample() + xmin_
    y = (ymax_ - ymin_) * random_state.random_sample() + ymin_
    z = 1.0
    vx = vy = vz = 0.0
    intial_state = np.array([x, y, z, vx, vy, vz])
    return intial_state


class Particle:
    def __init__(self, density, diameter, initial_state, bbox, max_depth, continent, step):
        self.id = counter.__next__()
        self.density = density
        self.diameter = diameter
        self.state = initial_state
        self.is_active = True
        self.step = step
        self.desintegrator = Desintegrator(bbox, max_depth, continent, step)

    def next_state(self, environment_func):
        """Get next state of the particle"""
        self.state = self._get_next_state(self.state, environment_func, self.density, self.step)
        logger.info(f'point %s next state is %s', self.id, self.state)
        if self.desintegrator.is_desintegrated(xyz=self.state[:3]):
            logger.info(f'point %s desintegrated', self.id)
            self.is_active = False

    def _get_next_state(self, state_prev, environment_func, density, step):

        coord_prev = state_prev[:3]
        velocity_prev = state_prev[3:6]
        coord = coord_prev + step * velocity_prev

        current_velocity = environment_func(coord)
        current_velocity = current_velocity.reshape(2)

        velocity = self._get_velocity(coord, density, current_velocity)
        states_cur = np.hstack([coord, velocity])
        return states_cur

    def _get_velocity(self, coord, density, current_velocity):
        """Get particle velocity"""
        if coord[2] == 1:
            if density <= 0.05:
                K = 4
            elif density <= 0.86:
                K = 1.3
            elif density <= 0.92:
                K = 1.25
            elif density <= 1.05:
                K = 1.15
            else:
                K = 1.0
        else:
            K = 1
        u, v = current_velocity * K

        d = self.diameter
        r = d / 2
        # плотность частицы
        # k = r * (((RO_F - r) / (RO_F - RO_W)) ** (1 / 3) - 1)
        k = 1.01
        kk = (r ** 3) / ((r + k) ** 3)
        ro_sphere = r * kk + RO_F * (1 - kk)
        # безразмерный диаметр частицы
        D = d * ((ro_sphere - RO_W) * g / RO_W / NU ** 2) ** (1 / 3)
        # скорость осаждения
        w = NU / d * D ** 3 * (38.1 + 0.93 * D ** (12 / 7)) ** (-7 / 8) / 100  # м/с

        return np.array([u, v, w])


class Desintegrator:
    def __init__(self, bbox, max_depth, continent, step):
        self.counter = 0.0
        self.bbox = bbox
        self.max_depth = max_depth
        self.continent = continent
        self.step = step

    def is_desintegrated(self, xyz):
        if not ((self.bbox[0] <= xyz[0] <= self.bbox[2]) and (self.bbox[1] <= xyz[1] <= self.bbox[3])):
            return True

        if self.continent.contains(xyz):
            return True

        if xyz[2] >= self.max_depth:
            return True

        self._count_point_is_near_continent(xyz)
        days = self.counter / (24 * 3600)
        if days >= 3:
            return True

        return False

    def _count_point_is_near_continent(self, xy):
        # Для каждой точки проверим попадает ли она в прибрежную зону
        is_near_continent = self.continent.buffer_contains(xy)
        if is_near_continent:
            # Запоминаем время пребывания в прибрежной зоне
            self.counter += 1 * self.step
        else:
            # Сбрасываем время пребывания в прибрежной зоне
            self.counter = 0.0
