import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


base_packages = ["numpy>=1.15.4", "scipy>=1.2.0", "scikit-learn>=0.20.2",
                 "pandas>=0.23.4", "matplotlib>=3.0.2", "torch>=1.0.0",
                 "plotnine>=0.5.1", "pytest==4.0.2", "jupyter==1.0.0", "jupyterlab==0.35.4"]

setup(
    name="scikit-blocks",
    version="0.0.1",
    packages=find_packages(exclude=['data', 'notebooks']),
    long_description=read('README.md'),
    install_requires=base_packages
)
