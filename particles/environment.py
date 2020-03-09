from math import floor
import logging

import numpy as np
import seawater
from scipy.interpolate.interpnd import LinearNDInterpolator

logger = logging.getLogger(__name__)


# for global cache
CACHE_INTERPOLATOR = {
    'last_day': 0,
    'last_interpolator': None,
}


def get_environment_func(seconds, current_df):
    # Создадим функцию интерполяции данных
    day = 1 + floor(seconds / 86400)
    if day > CACHE_INTERPOLATOR['last_day']:
        logger.info('Интерполируем течения')
        day_mask = current_df['time'] == day
        xyz = np.array(current_df.loc[day_mask, ['x', 'y', 'depth']])
        current_velocity = np.array(current_df.loc[day_mask, ['uo', 'vo']])
        interpolator = LinearNDInterpolator(xyz, current_velocity)
        CACHE_INTERPOLATOR['last_interpolator'] = interpolator
        CACHE_INTERPOLATOR['last_day'] = day
    else:
        interpolator = CACHE_INTERPOLATOR['last_interpolator']
    return interpolator
