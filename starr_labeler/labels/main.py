import hydra

from starr_labeler.labels.label_generator import label_generator


@hydra.main(version_base=None, config_path="disease_configs/")
def main(cfg):
    labels_class = label_generator(cfg)
    labels_class.compute_diagnosis_dates()
    labels_class.compute_diagnosis_labels()


main()
