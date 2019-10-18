import os
from setuptools import setup, find_packages

import sklego

base_packages = ["numpy>=1.15.4", "scipy>=1.2.0", "scikit-learn>=0.20.2",
                 "pandas>=0.23.4", "patsy>=0.5.1", "autograd>=1.2", "cvxpy>=1.0.24", "Deprecated>=1.2.6"]
docs_packages = ["sphinx>=1.8.5", "sphinx_rtd_theme>=0.4.3", "nbsphinx==0.4.2"]
dev_packages = docs_packages + ["flake8>=3.6.0", "matplotlib>=3.0.2", "pytest>=4.0.2", "nbval>=0.9.1",
                                "plotnine>=0.5.1", "jupyter>=1.0.0", "jupyterlab>=0.35.4", "pytest-cov>=2.6.1",
                                "pytest-mock>=1.6.3"]


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="scikit-lego",
    version=sklego.__version__,
    description='a collection of lego bricks for scikit-learn pipelines',
    author='Vincent D. Warmerdam & Matthijs Brouns',
    url="https://scikit-lego.readthedocs.io/en/latest/",
    packages=find_packages(exclude=['notebooks']),
    package_data={'sklego': ['data/*.csv']},
    long_description=read('readme.md'),
    long_description_content_type='text/markdown',
    install_requires=base_packages,
    extras_require={
        "docs": docs_packages,
        "dev": dev_packages
    },
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence']
)
