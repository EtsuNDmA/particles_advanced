import logging
from itertools import count

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

counter = count()


def get_current_data(file_name):
    """Get current vectors from file"""
    df = pd.read_csv(file_name)

    # replace invalid values with nan
    df[df.loc[:, ['uo', 'vo', 'so', 'thetao']] == 65535] = np.nan
    return df


def get_particles(file_name):
    """Get particles from file"""
    df = pd.read_csv(file_name)
    return df


def get_velocity(coord, density, current_velocity):
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
    return np.array([u, v, 0.0])


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


def get_next_state(state_prev, environment_func, density, step):
    """Get next state"""
    coord_prev = state_prev[:3]
    velocity_prev = state_prev[3:6]
    coord = coord_prev + step * velocity_prev

    current_velocity = environment_func(coord)
    current_velocity = current_velocity.reshape(2)

    velocity = get_velocity(coord, density, current_velocity)
    states_cur = np.hstack([coord, velocity])
    return states_cur


class Particle:
    def __init__(self, density, diameter, initial_state, bbox, continent, step):
        self.id = counter.__next__()
        self.density = density
        self.diameter = diameter
        self.state = initial_state
        self.is_active = True
        self.step = step
        self.desintegrator = Desintegrator(bbox, continent, step)

    def next_state(self, environment_func):
        self.state = get_next_state(self.state, environment_func, self.density, self.step)
        logger.debug(f'point %s next state is %s', self.id, self.state)
        if self.desintegrator.is_desintegrated(xy=self.state[:2]):
            logger.debug(f'point %s desintegrated', self.id)
            self.is_active = False


class Desintegrator:
    def __init__(self, bbox, continent, step):
        self.counter = 0.0
        self.bbox = bbox
        self.continent = continent
        self.step = step

    def is_desintegrated(self, xy):
        if not ((self.bbox[0] <= xy[0] <= self.bbox[2]) and (self.bbox[1] <= xy[1] <= self.bbox[3])):
            return True

        if self.continent.contains(xy):
            return True

        self._count_point_is_near_continent(xy)
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
