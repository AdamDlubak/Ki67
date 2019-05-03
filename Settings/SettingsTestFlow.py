import numpy as np

variables = {
    "data_folder": "../Data/Data Banknote Authentication.csv",
    "backup_folder": "Pickle/Data Banknote Authentication/",
    "results_folder": "Results/",
    "results_file": "Data Banknote Authentication - 5 Gausses.csv",

    "class_1": "One",
    "class_2": "Zero",
    
    "low": "Low",
    "middlelow": "Middle Low",
    "middle": "Middle",
    "middlehigh": "Middle High",
    "high": "High",
    
    "feature_numbers": 4,
    "set_min": 0,
    "set_max": 1,
    "fuzzy_sets_precision": 0.01,
    "show_results": False,
    "load_previous_data": False,

    "fuzzify_five": True
}

constraints = (slice(0, 1, 0.05), )
s_function_width = 5
sigma_mean_params = -1
n_folds = 10