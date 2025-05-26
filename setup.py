from setuptools import setup, find_packages

setup(
    name="grid_matching_tool",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "geopandas>=0.12.0",
        "pandas>=1.5.0",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "folium>=0.14.0",
        "pyyaml>=6.0",
        "shapely>=2.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for matching grid lines from different datasets",
    keywords="grid, energy, matching, gis",
    url="https://github.com/yourusername/grid-matching-tool",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)