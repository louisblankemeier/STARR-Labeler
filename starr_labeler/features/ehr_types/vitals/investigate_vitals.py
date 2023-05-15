import os

from starr_labeler.utils import data_iterator, frequency_by_num_patients, get_parser, setup_cfg

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    iterator = data_iterator(os.path.join(cfg.FEATURES.PATH, "clinical_data"), "flowsheets.csv")
    frequency_by_num_patients(iterator, "Measure", "./vitals_frequencies_by_patient.png", 30, 20)
