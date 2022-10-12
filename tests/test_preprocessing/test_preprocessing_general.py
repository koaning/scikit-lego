from sklego.common import TrainOnlyTransformerMixin

from sklego import preprocessing


def test_get_feature_names_out_implemented():
    for sklego_class in preprocessing.__all__:
        class_obj = getattr(preprocessing, sklego_class)
        # All preprocessors should have a get_feature_names_out method implemented, except for train only transformers.
        # Availability of get_feature_names_out is standard for all sklearn transformers since version 1.1.
        if not issubclass(class_obj, TrainOnlyTransformerMixin):
            assert hasattr(class_obj, "get_feature_names_out")
