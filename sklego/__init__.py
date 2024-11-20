import re
import sys

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata


__title__ = "sklego"
__version__ = metadata.version("scikit-lego")

SKLEARN_VERSION = tuple(int(re.sub(r"\D", "", str(v))) for v in metadata.version("scikit-learn").split("."))
