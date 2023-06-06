"""Configuration de la librarie"""
import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    description = fh.read()

setuptools.setup(
    name="causality-networks",
    version="1.0.0",
    author="Jean-Baptiste Delfau",
    author_email="jbdelfau@gmail.com",
    packages=setuptools.find_packages(exclude=("tests.*", "tests", "conf", "conf.*")),
    description="A library to build causality networks from discrete time series",
    long_description_content_type="text/markdown",
    url="https://github.com/jbdelfau/Causality-networks",
    license='GNU',
    python_requires='>=3.8',
    install_requires=[
        "matplotlib>=3.7.1",
        "networkx>=3.1",
        "numpy>=1.24.3",
        "scipy>=1.10.1"
    ],
)
