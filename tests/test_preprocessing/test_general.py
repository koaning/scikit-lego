from sklearn.utils.estimator_checks import (check_transformer_get_feature_names_out,
                                            check_transformer_get_feature_names_out_pandas)

from sklego import preprocessing
from sklego.common import TrainOnlyTransformerMixin


def test_get_feature_names_out_implemented():
    for sklego_class in preprocessing.__all__:
        class_obj = getattr(preprocessing, sklego_class)
        # All preprocessors should have a get_feature_names_out method implemented, except for train only transformers.
        # Availability of get_feature_names_out is standard for all sklearn transformers since version 1.1.
        if not issubclass(class_obj, TrainOnlyTransformerMixin):
            assert check_transformer_get_feature_names_out(name=sklego_class, transformer_orig=class_obj)
            assert check_transformer_get_feature_names_out_pandas(name=sklego_class, transformer_orig=class_obj)
