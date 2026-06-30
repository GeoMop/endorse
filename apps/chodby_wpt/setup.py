from setuptools import setup

setup(
    name="chodby_wpt",           # The name of your package
    version="0.1.0",             # Your package version
    description="Chodby project - water pressure test model.",
    author="Jan Stebel",
    author_email="jan.stebel@tul.cz",
    install_requires = ["PyYAML", "attrs", "numpy"],
    packages=["chodby_wpt"],
    package_dir={"chodby_wpt": "."},  # Map the package name to the current directory
)
