from setuptools import setup, find_packages

setup(
    name="grid-matcher",
    version="1.0.0",
    description="Tool for matching PyPSA transmission lines to JAO transmission lines",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "geopandas>=0.10.0",
        "shapely>=1.8.0",
        "networkx>=2.6.0",
        "folium>=0.12.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    entry_points={
        "console_scripts": [
            "grid-matcher=grid_matcher.cli:main",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)