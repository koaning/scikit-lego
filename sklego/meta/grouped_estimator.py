from deprecated import deprecated
from .grouped_predictor import GroupedPredictor


@deprecated(
    version="0.5.2",
    reason="Please use `sklego.meta.GroupedPredictor` instead. "
    "This object will be removed from the meta submodule in version 0.7.0.",
)
def GroupedEstimator(*args, **kwargs):
    return GroupedPredictor(*args, **kwargs)
