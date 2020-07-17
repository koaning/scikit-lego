VERSIONS = {"cvxpy": ">=1.0.24"}


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
        self.version = version if version else VERSIONS.get(package_name, "")

    def __getattr__(self, name):
        raise ImportError(
            f"The package {self.package_name}{self.version} is not installed"
        )
