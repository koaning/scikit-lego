VERSIONS = {"cvxpy": ">=1.0.24"}


class NotInstalledPackage:
    def __init__(self, package_name, version=None):
        self.package_name = package_name
        self.version = version if version else VERSIONS.get(package_name, "")

    def __getattr__(self, name):
        raise ImportError(
            f"The package {self.package_name}{self.version} is not installed"
        )
