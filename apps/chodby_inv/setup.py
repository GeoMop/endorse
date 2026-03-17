from setuptools import setup

setup(
    name="chodby_inv",           # The name of your package
    version="0.1.0",             # Your package version
    description="Chodby project - experiment inversion.",
    author="Jan Brezina",
    author_email="jan.brezina@tul.cz",
    install_requires = ["PyYAML", "attrs", "numpy"],
    packages=["chodby_inv"],
    package_dir={"chodby_inv": "."},  # Map the package name to the current directory
)
