from setuptools import setup, find_packages


base_packages = ["numpy>=1.15.4", "scipy>=1.2.0", "scikit-learn>=0.20.2",
                 "pandas>=0.23.4", "matplotlib>=3.0.2", "pytest==4.0.2",
                 "jupyter==1.0.0", "jupyterlab==0.35.4", "flake8==3.6.0"]

setup(
    name="scikit-lego",
    version="0.1.0",
    packages=find_packages(exclude=['notebooks']),
    install_requires=base_packages
)
