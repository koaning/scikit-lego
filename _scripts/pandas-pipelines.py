from pathlib import Path

_file = Path(__file__)
print(f"Executing {_file}")

_static_path = Path("_static") / _file.stem
_static_path.mkdir(parents=True, exist_ok=True)

import pandas as pd

# --8<-- [start:log-setup]
import logging

logging.basicConfig(level=logging.DEBUG)
# --8<-- [end:log-setup]

# --8<-- [start:data-setup]
from sklego.datasets import load_chicken

chickweight = load_chicken(as_frame=True)
# --8<-- [end:data-setup]

# --8<-- [start:log-step]
from sklego.pandas_utils import log_step

@log_step
def set_dtypes(chickweight):
    return chickweight.assign(
        diet=lambda d: d['diet'].astype('category'),
        chick=lambda d: d['chick'].astype('category'),
    )

chickweight.pipe(set_dtypes).head()
# --8<-- [end:log-step]

print(chickweight.pipe(set_dtypes).head())

# --8<-- [start:log-step-printfn]
@log_step(print_fn=logging.debug)
def remove_dead_chickens(chickweight):
    dead_chickens = chickweight.groupby('chick').size().loc[lambda s: s < 12]
    return chickweight.loc[lambda d: ~d['chick'].isin(dead_chickens)]


@log_step(print_fn=logging.info)
def remove_outliers(chickweight):
    return chickweight.pipe(remove_dead_chickens)

chickweight.pipe(set_dtypes).pipe(remove_outliers).head()
# --8<-- [end:log-step-printfn]

print(chickweight.pipe(set_dtypes).pipe(remove_outliers).head())

# --8<-- [start:log-step-notime]
@log_step(time_taken=False, shape=False, shape_delta=True)
def remove_dead_chickens(chickweight):
    dead_chickens = chickweight.groupby('chick').size().loc[lambda s: s < 12]
    return chickweight.loc[lambda d: ~d['chick'].isin(dead_chickens)]

chickweight.pipe(remove_dead_chickens).head()
# --8<-- [end:log-step-notime]

print(chickweight.pipe(remove_dead_chickens).head())


# --8<-- [start:log-step-extra]
from sklego.pandas_utils import log_step_extra

def count_unique_chicks(df, **kwargs):
    return "nchicks=" + str(df["chick"].nunique())

def display_message(df, msg):
    return msg


@log_step_extra(count_unique_chicks)
def start_pipe(df):
    """Get initial chick count"""
    return df


@log_step_extra(count_unique_chicks, display_message, msg="without diet 1")
def remove_diet_1_chicks(df):
    return df.loc[df["diet"] != 1]

(chickweight
 .pipe(start_pipe)
 .pipe(remove_diet_1_chicks)
 .head()
)
# --8<-- [end:log-step-extra]

print(chickweight.pipe(start_pipe).pipe(remove_diet_1_chicks).head())
