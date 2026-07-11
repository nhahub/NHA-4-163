#!/usr/bin/env python
# Setup script for Healthcare Hereditary Disease Prediction System
# Usage: python setup.py install

from setuptools import find_packages, setup

setup(
    name="healthcare-hereditary",
    version="0.1.0",
    description="Healthcare Hereditary Disease Prediction System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Healthcare Team",
    author_email="team@healthcare.local",
    url="https://github.com/your-org/healthcare-hereditary",
    license="Proprietary",
    python_requires=">=3.11",
    packages=find_packages(include=["libs*", "services*", "ml*", "pipelines*"]),
    install_requires=[
        # Web Framework & API
        "fastapi==0.110.3",
        "uvicorn[standard]==0.29.0",
        "pydantic==2.7.1",
        "pydantic-settings==2.2.1",
        # Data & ML
        "pandas==2.2.1",
        "numpy==1.26.4",
        "scikit-learn==1.4.2",
        "xgboost==2.0.3",
        # Web UI
        "streamlit==1.32.2",
        "plotly==5.18.0",
        # Databases
        "psycopg2-binary==2.9.9",
        "neo4j==5.19.0",
        "redis==5.0.4",
        # MLOps
        "mlflow==2.13.0",
        # Messaging
        "confluent-kafka==2.3.0",
        # Security
        "cryptography==42.0.5",
        # Database Migrations
        "alembic==1.13.1",
        # HTTP
        "requests==2.31.0",
        # Monitoring
        "prometheus-client==0.15.0",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.4",
            "pytest-cov==4.1.0",
            "pytest-asyncio==0.23.3",
            "pytest-mock==3.14.0",
            "ruff==0.3.5",
            "black==24.3.0",
            "mypy==1.9.0",
            "sphinx==7.3.7",
            "sphinx-rtd-theme==2.0.0",
            "ipython==8.22.2",
            "jupyter==1.0.0",
            "jupyterlab==4.1.5",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="healthcare ml disease prediction hereditary",
)
