variables = {
    "data_folder": "../Data/Haberman.csv",
    "backup_folder": "Pickle/Haberman/",
    "results_folder": "Results/",
    "results_file": "Haberman.csv",

    "class_1": "Zero",
    "class_2": "One",
    
    "low": "Low",
    "middlelow": "Middle Low",
    "middle": "Middle",
    "middlehigh": "Middle High",
    "high": "High",
    
    "feature_numbers": 3,
    "set_min": 0,
    "set_max": 1,
    "fuzzy_sets_precision": 0.01,
    "show_results": False,
    "load_previous_data": False,

    "fuzzify_five": False
}

constraints = (slice(0, 1, 0.05), )
s_function_width = 5
sigma_mean_params = -1
n_folds = 10