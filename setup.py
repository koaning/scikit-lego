import os
from setuptools import setup, find_packages

import sklego

base_packages = [
    "scikit-learn>=1.0",
    "pandas>=1.1.5",
    "Deprecated>=1.2.6",
]
cvxpy_packages = ["cvxpy>=1.1.8"]
umap_packages = ["umap-learn>=0.4.6"]
patsy_packages = ["patsy>=0.5.1"]
formulaic_packages = ["formulaic>=0.6.0"]
all_packages = cvxpy_packages + patsy_packages + formulaic_packages + umap_packages

docs_packages = [
    "sphinx==4.5.0",
    "sphinx_rtd_theme==1.0.0",
    "nbsphinx==0.8.8",
    "recommonmark==0.7.1"
]
test_packages = all_packages + [  # we need extras packages for their tests
    "flake8>=3.6.0",
    "nbval>=0.9.1",
    "pytest>=6.2.5",
    "pytest-xdist>=1.34.0",
    "black>=19.3b0",
    "pytest-cov>=2.6.1",
    "pytest-mock>=1.6.3",
    "pre-commit>=1.18.3",
]
util_packages = [
    "matplotlib>=3.0.2",
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
    packages=find_packages(exclude=["notebooks", "tests"]),
    package_data={"sklego": ["data/*.zip"]},
    long_description=read("readme.md"),
    long_description_content_type="text/markdown",
    install_requires=base_packages,
    extras_require={
        "base": base_packages,
        "cvxpy": cvxpy_packages,
        "umap": umap_packages,
        "patsy": patsy_packages,
        "formulaic": formulaic_packages,
        "all": all_packages,
        "docs": docs_packages,
        "dev": dev_packages,
        "test": test_packages,
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
