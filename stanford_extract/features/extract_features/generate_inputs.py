from pathlib import Path
import sys
from glob import glob
import hydra


path = Path(sys.path[0])
sys.path.insert(0, str(path.parent.parent.absolute()))
sys.path.insert(0, str(path.parent.absolute()))

from combine import compute_features, truncate
from utils import get_parser, setup_cfg

@hydra.main(version_base=None, config_path="/dataNAS/people/lblankem/opportunistic_ct/libraries/stanford_extract/configs", config_name = 'xgboost.yaml')
def main(cfg):
    truncate(cfg)
    #compute_features(cfg)

main()

    



