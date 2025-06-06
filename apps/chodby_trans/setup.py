from setuptools import setup

setup(
    name="chodby_trans",           # The name of your package
    version="0.1.0",             # Your package version
    description="Chodby project - transport model inversion.",
    author="Jan Brezina",
    author_email="jan.brezina@tul.cz",
    install_requires = ["PyYAML", "attrs", "numpy"],
    packages=["chodby_trans"],
    package_dir={"chodby_trans": "."},  # Map the package name to the current directory
)
