from pathlib import Path
import sys
from glob import glob
import hydra


path = Path(sys.path[0])
sys.path.insert(0, str(path.parent.parent.absolute()))
sys.path.insert(0, str(path.parent.absolute()))

from extract_features.combine import compute_features

@hydra.main(version_base=None, config_path="configs/", config_name = 'safe.yaml')
def main(cfg):
    compute_features(cfg)

main()

    



