import Settings.GeneralSettings as generalSettings

dataset = "Ki67"
class_1 = "brown"
class_2 = "other"

variables = {
    "dataset_name": dataset,
    "data_folder": "../Data/Ki67-Example/",
    "backup_folder": "Results/Ki67Test/",
    "results_folder": "Results/",
    "results_file": "Ki67Test.csv",
    "extension": ".png",

    "class_1": class_1,
    "class_2": class_2,
    "class_other": class_2,
    
    "feature_numbers": 9,
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