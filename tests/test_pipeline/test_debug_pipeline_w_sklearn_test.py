from sklearn import pipeline

from sklego.pipeline import DebugPipeline


pipeline.Pipeline = DebugPipeline


from sklearn.tests import test_pipeline  # noqa: E402


# In `test_set_pipeline_step_passthrough` the signature of the DebugPipeline
# includes a `log_callback`, which is not expected.
IGNORE_TESTS = ("test_set_pipeline_step_passthrough",)


for name, attr in test_pipeline.__dict__.items():
    if name not in globals() and name not in IGNORE_TESTS:
        globals()[name] = attr
