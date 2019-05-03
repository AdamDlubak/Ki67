import pickle

from Class.Reductor import Reductor as Reductor
from Class.Fuzzifier import Fuzzifier as Fuzzifier
from Class.ImageReader import ImageReader as ImageReader
from Class.RulesSetter import RulesSetter as RulesSetter
from Class.CSVExtractor import CSVExtractor as CSVExtractor
from Class.RulesExtractor import RulesExtractor as RulesExtractor
from Class.FeatureExtractor import FeatureExtractor as FeatureExtractor
from Class.InconsistenciesRemover import InconsistenciesRemover as InconsistenciesRemover

class Fuzzify(object):

    def fuzzifyFeatures(self, variables, sigma_mean_params = -1):
        fuzzifier = pickle.load(open(variables["backup_folder"] + "fuzzifier.p", "rb"))
        features_df = pickle.load(open(variables["backup_folder"] + "train_features_df.p", "rb"))

        if variables['fuzzify_five']:
            train_fuzzified_features_df, feature_labels, features, decision, fuzzify_parameters = fuzzifier.fuzzifyFive(features_df, sigma_mean_params)
        else:
            train_fuzzified_features_df, feature_labels, features, decision, fuzzify_parameters = fuzzifier.fuzzify(features_df, sigma_mean_params)
        
        pickle.dump(features, open(variables["backup_folder"] + "features.p", "wb"))
        pickle.dump(decision, open(variables["backup_folder"] + "decision.p", "wb"))
        pickle.dump(feature_labels, open(variables["backup_folder"] + "feature_labels.p", "wb"))
        pickle.dump(fuzzify_parameters, open(variables["backup_folder"] + "fuzzify_parameters.p", "wb"))
        pickle.dump(train_fuzzified_features_df, open(variables["backup_folder"] + "train_fuzzified_features_df.p", "wb"))
        
        inconsistencies_remover = InconsistenciesRemover(train_fuzzified_features_df, feature_labels, variables)
        decision_table = inconsistencies_remover.inconsistenciesRemoving()
        reductor = Reductor(decision_table, variables)
        decision_table_with_reduct = reductor.worker(decision_table)

        pickle.dump(reductor, open(variables["backup_folder"] + "reductor.p", "wb"))
        pickle.dump(decision_table_with_reduct, open(variables["backup_folder"] + "decision_table_with_reduct.p", "wb"))

    def worker(self, variables, sigma_mean_params = -1):
        self.fuzzifyFeatures(variables, sigma_mean_params)
