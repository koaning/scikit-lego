# Pandas pipelines

[Method chaining][method-chaining] is a great way for writing pandas code as it allows us to go from:

```py
raw_data = pd.read_parquet(...)
data_with_types = set_dtypes(raw_data)
data_without_outliers = remove_outliers(data_with_types)
```

to

```py
data = (
    pd.read_parquet(...)
    .pipe(set_dtypes)
    .pipe(remove_outliers)
)
```

But it does come at a cost, mostly in our ability to debug long pipelines. If there's a mistake somewhere along the way, you can only inspect the end result and lose the ability to inspect intermediate results.

A mitigation for this is to add _decorators_ to your pipeline functions that log common attributes of your dataframe on each step:

## Logging in method chaining

In order to use the logging capabilitites we first need to ensure we have a proper logger configured. We do this by running

```py
--8<-- "docs/_scripts/pandas-pipelines.py:log-setup"
```

Next load some data:

```py
--8<-- "docs/_scripts/pandas-pipelines.py:data-setup"
```

If we now add a [`log_step`][log-step-api] decorator to our pipeline function and execute the function, we see that we get some logging statements for free:

```py
--8<-- "docs/_scripts/pandas-pipelines.py:log-step"
```

```console

[set_dtypes(df)] time=0:00:00.015196 n_obs=578, n_col=4

   weight  time chick diet
0      42     0     1    1
1      51     2     1    1
2      59     4     1    1
3      64     6     1    1
4      76     8     1    1
```

We can choose to log at different log levels by changing the `print_fn` argument of the `log_step` decorator.

For example if we have a `remove_outliers` function that calls different outlier removal functions for different types of outliers, we might in general be only interested in the total outliers removed. In order to get that, we set the log level for our specific implementations to `logging.debug`:

```py
--8<-- "docs/_scripts/pandas-pipelines.py:log-step-printfn"
```

```console
DEBUG:root:[remove_dead_chickens(df)] time=0:00:00.005965 n_obs=519, n_col=4
INFO:root:[remove_outliers(df)] time=0:00:00.008321 n_obs=519, n_col=4
[set_dtypes(df)] time=0:00:00.001860 n_obs=578, n_col=4

   weight  time chick diet
0      42     0     1    1
1      51     2     1    1
2      59     4     1    1
3      64     6     1    1
4      76     8     1    1
```

The `log_step` function has some settings that let you tweak what exactly to log:

- `time_taken`: log the time it took to execute the function (default True)
- `shape`: log the output shape of the function (default True)
- `shape_delta`: log the difference in shape between input and output (default False)
- `names`: log the column names if the output (default False)
- `dtypes`: log the dtypes of the columns of the output (default False)

For example, if we don't care how long a function takes, but do want to see how many rows are removed if we remove dead chickens:

```py
--8<-- "docs/_scripts/pandas-pipelines.py:log-step-notime"
```

```console
[remove_dead_chickens(df)] delta=(-59, 0)

   weight  time  chick  diet
0      42     0      1     1
1      51     2      1     1
2      59     4      1     1
3      64     6      1     1
4      76     8      1     1
```

We can also define custom logging functions by using [`log_step_extra`][log-step-extra-api].

This takes any number of functions (> 1) that can take the output dataframe and return some output that can be converted to a string.

For example, if we want to log some arbitrary message and the number of unique chicks in our dataset, we can do:

```py
--8<-- "docs/_scripts/pandas-pipelines.py:log-step-extra"
```

```console
[start_pipe(df)] nchicks=50
[remove_diet_1_chicks(df)] nchicks=30 without diet 1

     weight  time  chick  diet
220      40     0     21     2
221      50     2     21     2
222      62     4     21     2
223      86     6     21     2
224     125     8     21     2
```

[log-step-api]: /api/pandas_pipeline#sklego.pandas_utils.log_step
[log-step-extra-api]: /api/pandas_pipeline#sklego.pandas_utils.log_step_extra
[method-chaining]: https://tomaugspurger.github.io/method-chaining
