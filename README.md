# Code for processing STARR electronic health record data
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/StanfordMIMI/Comp2Comp/ci.yaml?branch=master)

## Installation
```
git clone https://github.com/louisblankemeier/STARR-Labeler
cd STARR-Labeler
conda create --name starr_env
conda active starr_env
pip install -e .
```
## Generating Feature Vectors
First, create a config file in starr_labeler/features/configs. You can use the existing config files as templates. Use the following to generate features:
```
cd starr_labeler/features/
python main.py --config-name <name_of_config_file>
```

To run with slurm, edit the command in starr_labeler/features/slurm.py. Then, run:
```
python slurm.py
```

## Generating Outcome Labels
First, create a config file in starr_labeler/labels/disease_configs. You can use the existing config files as templates. Use the following to generate outcome labels:
```
cd starr_labeler/labels/
python main.py --config-name <name_of_config_file>
```

To run with slurm, edit the command in starr_labeler/labels/slurm.py. Then, run:
```
python slurm.py
```

## Citation
Please cite this work if you use it for your research.
