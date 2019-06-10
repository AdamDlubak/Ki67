class Settings(object):

    def __init__(self, generalSettings):

        self.class_1 = generalSettings.class_1
        self.file_name = generalSettings.file_name
        self.features_type = generalSettings.features_type




        self.adjustment_value = generalSettings.adjustment_value
        if self.adjustment_value == -1:
            self.adjustment = "Mean"
        elif self.adjustment_value == -2:
            self.adjustment = "Center"
        else:
            self.adjustment = "Optymalized"

        self.class_2 = "other"
        self.extension = ".png"

        self.dataset = "Ki67 " + str(generalSettings.gausses) + " " + self.adjustment + " " + generalSettings.style + " " + generalSettings.class_1 + " " + str(generalSettings.features_type)

        self.dataset_name = self.dataset
        self.data_folder_train = generalSettings.data_folder + "Ki67-Train/"
        self.data_folder_test = generalSettings.data_folder + "Ki67-Test/"    
        self.data_folder_veryfication = generalSettings.data_folder + "Ki67-Veryfication/"       

        
        self.backup_folder = generalSettings.backup_folder + self.dataset + "/"

        self.results_folder = generalSettings.results_folder
        self.results_file = self.dataset + ".csv"
        
        self.feature_numbers = 2
        self.gausses = generalSettings.gausses
        self.test_type = generalSettings.test_type
        self.style = generalSettings.style

        self.is_training = generalSettings.is_training
        self.default_s_value = generalSettings.default_s_value

        self.set_min = 0
        self.set_max = 1
        self.fuzzy_sets_precision = 0.01
        self.show_results = generalSettings.show_results
        self.load_previous_data = False

        self.verylow = "Very low"
        self.low = "Low"
        self.middlelowminus = "Middle Low -"
        self.middlelow = "Middle Low"
        self.middlelowplus = "Middle Low +"
        self.middle = "Middle"
        self.middlehighminus = "Middle High -"
        self.middlehigh = "Middle High"
        self.middlehighplus = "Middle High +"
        self.high = "High"
        self.veryhigh = "Very High"

        self.constraints = (slice(0.2, 0.8, 0.05), )
        self.constraints_adj = (slice(0.3, 0.7, 0.1), slice(0.3, 0.7, 0.1), slice(0.3, 0.7, 0.1), slice(0.3, 0.7, 0.1), slice(0.3, 0.7, 0.1), slice(0.3, 0.7, 0.1), slice(0.3, 0.7, 0.1), )

        self.s_function_width = generalSettings.s_function_width
        self.n_folds = 5