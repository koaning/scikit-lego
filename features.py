import inspect
from sklego import meta, pipeline, pandas_utils, dummy, linear_model, mixture, \
    naive_bayes, datasets, model_selection, preprocessing, metrics


def not_in(thing, *substrings):
    for string in substrings:
        if string in thing:
            return False
    return True


def print_classes(submodule):
    for cls in dir(submodule):
        if inspect.isclass(getattr(submodule, cls)):
            if not_in(cls, 'Mixin', 'Base'):
                if (cls[0].upper() == cls[0]) and (cls[0] != '_'):
                    print(f"{submodule.__name__}.{cls}")


def print_functions(submodule):
    for cls in dir(submodule):
        if inspect.isfunction(getattr(submodule, cls)):
            if cls[0] != '_':
                print(f"{submodule.__name__}.{cls}")


if __name__ == "__main__":
    print_functions(datasets)
    print_functions(pandas_utils)
    print_classes(dummy)
    print_classes(linear_model)
    print_classes(naive_bayes)
    print_classes(mixture)
    print_classes(meta)
    print_classes(preprocessing)
    print_classes(model_selection)
    print_classes(pipeline)
    print_functions(metrics)
