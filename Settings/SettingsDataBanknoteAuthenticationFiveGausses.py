import numpy as np

variables = {
    "data_folder": "../Data/Data Banknote Authentication.csv",
    "backup_folder": "1. Data - Data Banknote Authentication - 5 Gausses/",
    "results_folder": "1. Results - Data Banknote Authentication - 5 Gausses/",
    "extension": ".jpeg",
    
    "d_low": "One",
    "d_middle": "",
    "d_high": "Zero",
    
    "middlelow": "Middle Low",
    "low": "Low",
    "middle": "Middle",
    "high": "High",
    "middlehigh": "Middle High",
    
    "feature_numbers": 4,
    "set_min": 0,
    "set_max": 1,
    "fuzzy_sets_precision": 0.01,
    "show_results": False,
    "load_previous_data": False,

    "pairplot_data_file": "all_features_table",
    "fuzzify_five": True
}

constraints = ((0, 1, 0.01),)
sigma_mean_params = -1
n_folds = 10

swarm_size = 20
dim = 1
epsilon = 1.0
iters = 20
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
borders = (np.array([0]), np.array([1]))

threshold_value = -1 