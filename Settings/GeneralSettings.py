# ---- Modes ----
# 0 - Fuzzification
# 1 - Training Set - K Fold + Test
# 2 - Training Set
# 3 - Test Set
# 4 - Training Set + Test Set

mode = 4                                  # 1 / 2 / 3 / 4
gausses = 9                              # 3 / 5 / 7 / 9 / 11 

class_1 = "blue"
file_name = "Testowe zdjęcie"
features_type = 1

test_type = "Ki67 Segmentation"       # Gauss Style / Gauss Number / Gauss Adjustment / S-Function Adjustment
style = "Gaussian Equal"                  # Gaussian Equal / Gaussian Progressive
adjustment_value = -1                     # Center == -2 / Mean == -1 / Optymalized == Value
default_s_value = 0.5                       
is_training = True                        # True / False
show_results = False                      # True / False
data_folder = "../Data/"
backup_folder = "Pickle/"
results_folder = "Results/"

s_function_width = 5