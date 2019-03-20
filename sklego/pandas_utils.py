import inspect
from functools import wraps
import datetime as dt
import logging


def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(inspect.stack()[1].function)

        tic = dt.datetime.now()
        result_df = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)

        logger.info(f"[{func.__name__}]  n_obs={result_df.shape[0]} n_col={result_df.shape[1]} time={time_taken}")
        return result_df
    return wrapper
