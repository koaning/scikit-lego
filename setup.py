import os
from setuptools import setup, find_packages

import sklego

base_packages = [
    "scikit-learn>=0.24.1",
    "pandas>=1.1.5",
    "patsy>=0.5.1",
    "autograd>=1.2",
    "Deprecated>=1.2.6",
    "umap-learn>=0.4.6"
]
cvxpy_packages = ["cvxpy>=1.1.8"]
all_packages = cvxpy_packages

docs_packages = [
    "sphinx==1.8.5",
    "sphinx_rtd_theme>=0.4.3",
    "nbsphinx>=0.4.2",
    "recommonmark==0.6.0",
]
test_packages = all_packages + [  # we need extras packages for their tests
    "flake8>=3.6.0",
    "nbval>=0.9.1",
    "pytest==5.4.1",
    "pytest-xdist==1.34.0",
    "black>=19.3b0",
    "pytest-cov>=2.6.1",
    "pytest-mock>=1.6.3",
    "pre-commit>=1.18.3",
]
util_packages = [
    "matplotlib>=3.0.2",
    "plotnine>=0.5.1",
    "jupyter>=1.0.0",
    "jupyterlab>=0.35.4",
]
dev_packages = docs_packages + test_packages + util_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="scikit-lego",
    version=sklego.__version__,
    description="a collection of lego bricks for scikit-learn pipelines",
    author="Vincent D. Warmerdam & Matthijs Brouns",
    url="https://scikit-lego.netlify.app/",
    packages=find_packages(exclude=["notebooks"]),
    package_data={"sklego": ["data/*.zip"]},
    long_description=read("readme.md"),
    long_description_content_type="text/markdown",
    install_requires=base_packages,
    extras_require={
        "base": base_packages,
        "cvxpy": cvxpy_packages,
        "all": all_packages,
        "docs": docs_packages,
        "dev": dev_packages,
        "test": test_packages,
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
