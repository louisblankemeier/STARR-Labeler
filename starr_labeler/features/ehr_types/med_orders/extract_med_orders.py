from starr_labeler.features.extract_features import extract_base


class extract_med_orders(extract_base):
    def __init__(self, config, file_name, feature_type):
        super().__init__(config, file_name, feature_type)

    def process_data(self, pat_data):
        # med_orders_regex = "|".join(list(self.cfg['FEATURES']['TYPES']['MED_ORDERS']['INCLUDE'].keys()))
        # pat_data = pat_data.loc[pat_data['Sig'].str.contains(med_orders_regex, regex = True,
        # case = False, na = False)]
        # pat_data.loc[:, 'Type'] = 'Blood Pressure'
        pat_data.loc[:, "Value"] = 1
        # pat_data = pat_data[['Patient Id', 'Medication', 'Value', 'Start Date']]
        pat_data = pat_data[["Patient Id", "Therapeutic Class", "Value", "Start Date"]]
        # pat_data = pat_data[['Patient Id', 'Type', 'Value', 'Start Date']]
        pat_data.columns = ["Patient Id", "Type", "Value", "Event_dt"]
        return pat_data

    def truncate_data(self, pat_data):
        return pat_data
