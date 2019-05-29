class Settings(object):

    def __init__(self, generalSettings):
        self.dataset = "Haberman"
        self.class_1 = "One"
        self.class_2 = "Zero"

        self.dataset_name = self.dataset
        self.data_folder = generalSettings.data_folder + self.dataset + ".csv"
        self.backup_folder = generalSettings.backup_folder + self.dataset + "/" + str(generalSettings.gausses) + " Gausses/"

        self.results_folder = generalSettings.results_folder
        self.results_file = self.dataset + ".csv"
        
        self.feature_numbers = 3
        self.gausses = generalSettings.gausses
        self.test_type = generalSettings.test_type
        self.style = generalSettings.style
        self.adjustment_value = generalSettings.adjustment_value
        if self.adjustment_value == -1:
            self.adjustment = "Mean"
        elif self.adjustment_value == 0:
            self.adjustment = "Center"
        else:
            self.adjustment = "Optymalized"
        self.is_training = generalSettings.is_training
        self.default_s_value = generalSettings.default_s_value

        self.set_min = 0
        self.set_max = 1
        self.fuzzy_sets_precision = 0.001
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
        self.s_function_width = generalSettings.s_function_width
        self.n_folds = 5