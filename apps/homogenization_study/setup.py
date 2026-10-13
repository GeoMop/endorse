"""Package configuration for the homogenization study application."""

from setuptools import setup


setup(
    name="homogenization_study",
    version="0.1.0",
    description="Monte Carlo study of homogenized conductivity tensor fields.",
    author="Jan Brezina",
    author_email="jan.brezina@tul.cz",
    install_requires=[
        "PyYAML",
        "attrs",
        "numpy<2",
        "dask",
        "distributed",
        "salib",
        "openturns",
        "joblib",
        "loky",
        "matplotlib",
        "zarr_fuse @ git+https://github.com/GeoMop/zarr_fuse.git@main",
    ],
    packages=["homogenization_study"],
    package_dir={"homogenization_study": "."},
)
