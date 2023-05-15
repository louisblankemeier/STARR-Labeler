import hydra

from starr_labeler.features.extract_features.combine import compute_features


@hydra.main(version_base=None, config_path="configs/")
def main(cfg):
    compute_features(cfg)


main()
