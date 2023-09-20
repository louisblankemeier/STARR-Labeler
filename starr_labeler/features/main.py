import hydra

from starr_labeler.features.extract_features import combine


@hydra.main(version_base=None, config_path="configs/")
def main(cfg):
    combine.compute_features(cfg)


main()
