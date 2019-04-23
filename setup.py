from setuptools import setup, find_packages
import os

base_packages = ["numpy>=1.15.4", "scipy>=1.2.0", "scikit-learn>=0.20.2",
                 "pandas>=0.23.4", "patsy>=0.5.1", "autograd>=1.2"]

docs_packages = ["sphinx>=1.8.5", "sphinx_rtd_theme>=0.4.3"]
dev_packages = docs_packages + ["flake8>=3.6.0", "matplotlib>=3.0.2", "pytest>=4.0.2",
                                "nbval>=0.9.1", "plotnine>=0.5.1", "jupyter>=1.0.0",
                                "jupyterlab>=0.35.4"]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="scikit-lego",
    version="0.1.5",
    packages=find_packages(exclude=['notebooks']),
    long_description=read('readme.md'),
    long_description_content_type='text/markdown',
    install_requires=base_packages,
    extras_require={
        "docs": docs_packages,
        "dev": dev_packages
    }
)
