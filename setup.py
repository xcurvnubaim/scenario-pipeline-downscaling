"""Package setup for scenario-pipeline-downscaling."""

from setuptools import find_packages, setup

setup(
    name="scenario-pipeline-downscaling",
    version="0.1.0",
    description="Modular PyTorch pipeline for spatiotemporal climate downscaling",
    packages=find_packages(where=".", include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "xarray>=2023.1.0",
        "zarr>=2.14.0",
        "dask[array]>=2023.1.0",
        "numpy>=1.24.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "mlflow>=2.4.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "geo": ["cartopy>=0.21.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
)
