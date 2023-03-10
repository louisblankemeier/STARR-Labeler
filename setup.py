from setuptools import setup, find_packages

setup(name='STARR-Labeler',
version='0.0.1',
author='Louis Blankemeier',
description='Transform raw electronic health records data into features and labels.',
packages=find_packages(),
install_requires=[
        'pandas',
        'hydra-core',
        'torch',
        'transformers',
        'numpy',
        'regex',
        'matplotlib',
    ],
)