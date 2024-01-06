# Debug pipeline

This document demonstrates how you might use a [`DebugPipeline`][debug-pipe-api]. It is much like a normal scikit-learn [`Pipeline`][pipe-api] but it offers more debugging options.

We'll first set up libraries and config.

```py title="Setup"
--8<-- "docs/_scripts/debug-pipeline.py:setup"
```

Next up, let's make a simple transformer.

```py title="Simple transformer"
--8<-- "docs/_scripts/debug-pipeline.py:simple-pipe"
```

This pipeline behaves exactly the same as a normal pipeline. So let's use it.

```py title="Simple transformer"
--8<-- "docs/_scripts/debug-pipeline.py:simple-pipe-fit-transform"
```

```console
Transformed X:
 [[1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111.]]
```

## Log statements

It is possible to set a `log_callback` variable that logs in between each step.

!!! note
    There are _three_ log statements while there are _four_ steps, because there are _three_ moments _in between_ the steps.
    The output can be checked outside of the pipeline.

```py title="'default' log_callback"
--8<-- "docs/_scripts/debug-pipeline.py:log-callback"
```

```console
[default_log_callback:38] - [Adder(value=1)] shape=(3, 5) time=0s
[default_log_callback:38] - [Adder(value=10)] shape=(3, 5) time=0s
[default_log_callback:38] - [Adder(value=100)] shape=(3, 5) time=0s
Transformed X:
 [[1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111.]]
```

## Set the `log_callback` function later

It is possible to set the `log_callback` later then initialisation.

```py title="log_callback after initialisation"
--8<-- "docs/_scripts/debug-pipeline.py:log-callback-after"
```

```console
[default_log_callback:38] - [Adder(value=1)] shape=(3, 5) time=0s
[default_log_callback:38] - [Adder(value=10)] shape=(3, 5) time=0s
[default_log_callback:38] - [Adder(value=100)] shape=(3, 5) time=0s
Transformed X:
 [[1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111.]]
```

## Custom `log_callback`

The custom log callback function expect the output of each step, which is an tuple containing the output of the step and the step itself, and the execution time of the step.

```py title="Custom log_callback"
--8<-- "docs/_scripts/debug-pipeline.py:custom-log-callback"
```

```console
[log_callback:16] - [Adder(value=1)] shape=(3, 5) nbytes=120 time=5.340576171875e-05
[log_callback:16] - [Adder(value=10)] shape=(3, 5) nbytes=120 time=6.651878356933594e-05
[log_callback:16] - [Adder(value=100)] shape=(3, 5) nbytes=120 time=6.723403930664062e-05
Transformed X:
 [[1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111.]]
```

## Feature union

Feature union also works with the debug pipeline.

```py title="Feature union"
--8<-- "docs/_scripts/debug-pipeline.py:feature-union"
```

```console
[default_log_callback:38] - [Adder(value=1)] shape=(3, 5) time=0s
[default_log_callback:38] - [Adder(value=10)] shape=(3, 5) time=0s
[default_log_callback:38] - [Adder(value=100)] shape=(3, 5) time=0s
[log_callback:16] - [Adder(value=1)] shape=(3, 5) nbytes=120 time=4.482269287109375e-05
[log_callback:16] - [Adder(value=10)] shape=(3, 5) nbytes=120 time=5.1021575927734375e-05
[log_callback:16] - [Adder(value=100)] shape=(3, 5) nbytes=120 time=6.365776062011719e-05
Transformed X:
 [[1111. 1111. 1111. 1111. 1111. 1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111. 1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111. 1111. 1111. 1111. 1111. 1111.]]
```

## Enough logging

Remove the `log_callback` function when not needed anymore.

```py title="Remove log_callback"
--8<-- "docs/_scripts/debug-pipeline.py:remove"
```

```console
Transformed X:
 [[1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111.]
 [1111. 1111. 1111. 1111. 1111.]]
```

[debug-pipe-api]: ../../api/pipeline#sklego.pipeline.DebugPipeline
[pipe-api]: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
