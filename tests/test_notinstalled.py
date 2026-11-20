from importlib.metadata import PackageNotFoundError, requires

import pytest
from packaging.requirements import Requirement

from sklego.notinstalled import KNOWN_PACKAGES


def test_installed_package_works():
    package_name = "pandas"
    version = ">=0.23.4"

    try:
        import pandas as pd
    except ImportError:
        from sklego.notinstalled import NotInstalledPackage

        pd = NotInstalledPackage(package_name, version=version)

    assert pd.__version__


def test_uninstsalled_package_raises():
    package_name = "thispackagedoesnotexist"
    version = "==1.2.3"

    try:
        import thispackagedoesnotexist as package
    except ImportError:
        from sklego.notinstalled import NotInstalledPackage

        package = NotInstalledPackage(package_name, version=version)

    with pytest.raises(ImportError) as e:
        package.__version__
    assert package_name in str(e.value)
    assert version in str(e.value)


@pytest.mark.parametrize("package_name", list(KNOWN_PACKAGES))
def test_known_packages_match_optional_dependencies(package_name):
    """The versions in KNOWN_PACKAGES should match the extras declared in pyproject.toml."""
    try:
        requirements = [Requirement(req) for req in requires("scikit-lego") or []]
    except PackageNotFoundError:
        pytest.skip("scikit-lego is not installed, so its metadata is not available")

    package_info = KNOWN_PACKAGES[package_name]
    extra_requirements = [
        req
        for req in requirements
        if req.name == package_name and req.marker and req.marker.evaluate({"extra": package_info["extra_name"]})
    ]

    assert len(extra_requirements) == 1
    assert str(extra_requirements[0].specifier) == package_info["version"]
