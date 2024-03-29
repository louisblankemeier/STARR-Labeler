from setuptools import find_packages, setup

setup(
    name="STARR-Labeler",
    version="0.0.1",
    author="Louis Blankemeier",
    description="Transform raw electronic health records data into features and labels.",
    packages=find_packages(),
    install_requires=[
        "pandas==1.3.2",
        "hydra-core",
        "numpy==1.20.3",
        "regex",
        "matplotlib",
        "tqdm",
        "scikit-learn",
        "tabulate",
    ],
    extras_require={
        "ml": [
            "torch",
            "transformers",
        ],
    },
)
