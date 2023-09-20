import hydra

from starr_labeler.labels import label_generator


@hydra.main(version_base=None, config_path="disease_configs/")
def main(cfg):
    labels_class = label_generator.LabelGenerator(cfg)
    labels_class.compute_diagnosis_dates()
    labels_class.compute_diagnosis_labels()


main()
