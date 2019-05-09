import pickle
from sklearn.model_selection import train_test_split

from Class.Reductor import Reductor as Reductor
from Class.Fuzzifier import Fuzzifier as Fuzzifier
from Class.ImageReader import ImageReader as ImageReader
from Class.RulesSetter import RulesSetter as RulesSetter
from Class.CSVExtractor import CSVExtractor as CSVExtractor
from Class.RulesExtractor import RulesExtractor as RulesExtractor
from Class.FeatureExtractor import FeatureExtractor as FeatureExtractor
from Class.InconsistenciesRemover import InconsistenciesRemover as InconsistenciesRemover

class Fuzzify(object):

    def __init__(self):
        self.variables = []

    def fuzzifyFeatures(self, sigma_mean_params = -1):
        fuzzifier = pickle.load(open(self.variables["backup_folder"] + "fuzzifier.p", "rb"))
        features_df = pickle.load(open(self.variables["backup_folder"] + "train_features_df.p", "rb"))

        train_fuzzified_features_df, feature_labels, features, decision, fuzzify_parameters = fuzzifier.fuzzify(features_df, sigma_mean_params)
        
        pickle.dump(features, open(self.variables["backup_folder"] + "features.p", "wb"))
        pickle.dump(decision, open(self.variables["backup_folder"] + "decision.p", "wb"))
        pickle.dump(feature_labels, open(self.variables["backup_folder"] + "feature_labels.p", "wb"))
        pickle.dump(fuzzify_parameters, open(self.variables["backup_folder"] + "fuzzify_parameters.p", "wb"))
        pickle.dump(train_fuzzified_features_df, open(self.variables["backup_folder"] + "train_fuzzified_features_df.p", "wb"))

        inconsistencies_remover = InconsistenciesRemover(train_fuzzified_features_df, feature_labels, self.variables)
        decision_table, changed_decisions = inconsistencies_remover.inconsistenciesRemoving()
        reductor = Reductor(decision_table, self.variables)
        decision_table_with_reduct, features_number_after_reduct = reductor.worker(decision_table)

        pickle.dump(reductor, open(self.variables["backup_folder"] + "reductor.p", "wb"))
        pickle.dump(decision_table_with_reduct, open(self.variables["backup_folder"] + "decision_table_with_reduct.p", "wb"))

        implicants_number = decision_table_with_reduct.shape[0]
        return changed_decisions, features_number_after_reduct, implicants_number, fuzzify_parameters

    def worker(self, variables, searched_class, sigma_mean_params):
        self.variables = variables

        changed_decisions, features_number_after_reduct, implicants_number, fuzzify_parameters = self.fuzzifyFeatures(sigma_mean_params)
        return changed_decisions, features_number_after_reduct, implicants_number, fuzzify_parameters
