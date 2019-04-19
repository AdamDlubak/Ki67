import numpy as np

variables = {
    "data_folder": "../Data/Baberman.csv",
    "backup_folder": "1. Data - Baberman/",
    "results_folder": "1. Results - Baberman/",
    "extension": ".jpeg",
    
    "d_low": "One",
    "d_middle": "",
    "d_high": "Zero",
    
    "verylow": "Very Low",
    "low": "Low",
    "middle": "Middle",
    "high": "High",
    "veryhigh": "Very High",
    
    "feature_numbers": 3,
    "set_min": 0,
    "set_max": 1,
    "fuzzy_sets_precision": 0.01,
    "show_results": False,
    "load_previous_data": False,

    "pairplot_data_file": "all_features_table",
    "fuzzify_five": False
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