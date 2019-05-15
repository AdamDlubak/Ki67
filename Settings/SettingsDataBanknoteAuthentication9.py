import Settings.GeneralSettings9 as generalSettings

dataset = "Data Banknote Authentication"
class_1 = "One"
class_2 = "Zero"

variables = {
    "dataset_name": dataset,
    "data_folder": generalSettings.data_folder + dataset + ".csv",
    "backup_folder": generalSettings.backup_folder + dataset + "/" + str(generalSettings.gausses) + " Gausses/",
    "results_folder": generalSettings.results_folder,
    "results_file": dataset + ".csv",

    "class_1": class_1,
    "class_2": class_2,
    
    "feature_numbers": 4,
    "gausses": generalSettings.gausses,
    "test_type": generalSettings.test_type,

    "set_min": 0,
    "set_max": 1,
    "fuzzy_sets_precision": 0.01,
    "show_results": False,
    "load_previous_data": False,

    "verylow": "Very low",
    "low": "Low",
    "middlelowminus": "Middle Low -",
    "middlelow": "Middle Low",
    "middlelowplus": "Middle Low +",
    "middle": "Middle",
    "middlehighminus": "Middle High -",
    "middlehigh": "Middle High",
    "middlehighplus": "Middle High +",
    "high": "High",
    "veryhigh": "Very High",
}

constraints = (slice(0, 1, 0.05), )
s_function_width = generalSettings.s_function_width
sigma_mean_params = generalSettings.sigma_mean_params
n_folds = 10