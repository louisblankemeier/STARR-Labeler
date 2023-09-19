import pandas as pd

from starr_labeler.features.extract_features.extract import extract_base


class extract_demographics(extract_base):
    def __init__(self, config, file_name, feature_type):
        super().__init__(config, file_name, feature_type)

    def process_data(self, pat_data, first_call=False):
        demographics_regex = "|".join(
            list(self.cfg["FEATURES"]["TYPES"]["DEMOGRAPHICS"]["INCLUDE"].keys())
        )
        demographics_regex += "|Patient Id"
        pat_data = pat_data.loc[
            :,
            list(
                pd.Series(pat_data.columns).str.contains(
                    demographics_regex, regex=True, case=False, na=False
                )
            ),
        ]
        if "Date of Birth" in pat_data.columns:
            pat_data.loc[:, "Date of Birth"] = pd.to_datetime(
                pat_data.loc[:, "Date of Birth"], utc=True
            )

        if "Race" in pat_data.columns:
            original_categories_race = pat_data.loc[:, "Race"].astype("category")
            pat_data.loc[:, "Race"] = original_categories_race.cat.codes
            if first_call:
                race_mapping = dict(enumerate(original_categories_race.cat.categories))
                print("Race mappings:", race_mapping)

        if "Gender" in pat_data.columns:
            original_categories_gender = pat_data.loc[:, "Gender"].astype("category")
            pat_data.loc[:, "Gender"] = original_categories_gender.cat.codes
            if first_call:
                gender_mapping = dict(enumerate(original_categories_gender.cat.categories))
                print("Gender mappings:", gender_mapping)

        if "Ethnicity" in pat_data.columns:
            original_categories_ethnicity = pat_data.loc[:, "Ethnicity"].astype(
                "category"
            )
            pat_data.loc[:, "Ethnicity"] = original_categories_ethnicity.cat.codes
            if first_call:
                ethnicity_mapping = dict(
                    enumerate(original_categories_ethnicity.cat.categories)
                )
                print("Ethnicity mappings:", ethnicity_mapping)

        return pat_data

    def truncate_data(self, pat_data):
        demographics_regex = "|".join(
            list(self.cfg["FEATURES"]["TYPES"]["DEMOGRAPHICS"]["INCLUDE"].keys())
        )
        demographics_regex += "|Patient Id"
        pat_data = pat_data.loc[
            :,
            list(
                pd.Series(pat_data.columns).str.contains(
                    demographics_regex, regex=True, case=False, na=False
                )
            ),
        ]
        truncated = pat_data
        truncated = truncated.sort_values(by=["Patient Id"])
        return truncated
