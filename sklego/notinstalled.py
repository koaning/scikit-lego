KNOWN_PACKAGES = {"cvxpy": {"version": ">=1.0.24", "extra_name": "cvxpy"}}


class NotInstalledPackage:
    """
    Class to gracefully catch ImportErrors for modules and packages that are not installed

    :param package_name (str): Name of the package you want to load
    :param version (str, Optional): Version of the package

    Usage:
        >>> try:
        ...     import thispackagedoesnotexist as package
        >>> except ImportError:
        ...     from sklego.notinstalled import NotInstalledPackage
        ...     package = NotInstalledPackage("thispackagedoesnotexist")
    """

    def __init__(self, package_name: str, version: str = None):
        self.package_name = package_name
        package_info = KNOWN_PACKAGES.get(package_name, {})
        self.version = version if version else package_info.get("version", "")

        extra_name = package_info.get("extra_name", None)
        self.pip_message = (
            (
                f"Install extra requirement {package_name} using "
                + f"`python -m pip install scikit-lego[{extra_name}]` or "
                + "`python -m pip install scikit-lego[all]`. "
                + "For more information, check the 'Dependency installs' section of the installation docs at "
                + "https://scikit-lego.readthedocs.io/en/latest/install.html"
            )
            if extra_name
            else ""
        )

    def __getattr__(self, name):
        raise ImportError(
            f"The package {self.package_name}{self.version} is not installed. "
            + self.pip_message
        )
