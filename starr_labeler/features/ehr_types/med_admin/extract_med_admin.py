from starr_labeler.features.extract_features.extract import extract_base


class extract_med_admin(extract_base):
    def __init__(self, config, file_name, feature_type):
        super().__init__(config, file_name, feature_type)

    def process_data(self, pat_data):
        pat_data.loc[:, "Value"] = 1
        pat_data = pat_data[["Patient Id", "Medication", "Value", "Taken Date"]]
        pat_data.columns = ["Patient Id", "Type", "Value", "Event_dt"]
        return pat_data

    def truncate_data(self, pat_data):
        return pat_data
