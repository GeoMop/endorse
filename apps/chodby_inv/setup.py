from setuptools import setup

setup(
    name="chodby_inv",           # The name of your package
    version="0.1.0",             # Your package version
    description="A super flat layout package for demonstration.",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires = ["PinyYAML", "attrs", "numpy"],
    packages=["chodby_inv"],
    package_dir={"chodby_inv": "."},  # Map the package name to the current directory
)
