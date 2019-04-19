import os
import pickle
from sklearn.model_selection import train_test_split

from Functions.Reductor import Reductor as Reductor
from Functions.Fuzzifier import Fuzzifier as Fuzzifier
from Functions.ImageReader import ImageReader as ImageReader
from Functions.RulesSetter import RulesSetter as RulesSetter
from Functions.CSVExtractor import CSVExtractor as CSVExtractor
from Functions.RulesExtractor import RulesExtractor as RulesExtractor
from Functions.FeatureExtractor import FeatureExtractor as FeatureExtractor
from Functions.InconsistenciesRemover import InconsistenciesRemover as InconsistenciesRemover

class LoadCSV(object):

    def saveVariables(self, variables):
        d_results = [variables["d_low"], variables["d_middle"], variables["d_high"]]
        levels = [variables["low"], variables["middle"], variables["high"]]

        if not os.path.exists(variables["backup_folder"]):
            os.makedirs(variables["backup_folder"])
        if not os.path.exists(variables["results_folder"]):
            os.makedirs(variables["results_folder"])   

        pickle.dump([variables, d_results, levels], open(variables["backup_folder"] + "parameters.p", "wb"))

        return variables, d_results, levels

    def prepareData(self, variables, levels, d_results):

        if variables['load_previous_data']:
            fuzzifier = pickle.load(open(variables["backup_folder"] + "fuzzifier.p", "rb"))
            normalized_features_table = pickle.load(open(variables["backup_folder"] + "normalized_features_table.p", "rb"))

        else:
            fuzzifier = Fuzzifier(variables, levels, d_results)
            
            csvExtractor = CSVExtractor(variables, fuzzifier)
            all_normalized_features_table, all_features_table = csvExtractor.worker()
            
            pickle.dump(fuzzifier, open(variables["backup_folder"] + "fuzzifier.p", "wb"))
            pickle.dump(all_normalized_features_table, open(variables["backup_folder"] + "all_normalized_features_table.p", "wb"))
            pickle.dump(all_features_table, open(variables["backup_folder"] + "all_features_table.p", "wb"))
        
        return fuzzifier, all_normalized_features_table

    def splitDataForTrainingTest(self, features_table, variables, test_size = 0.2):
        train_normalized_features_table, test_normalized_features_table = train_test_split(features_table, test_size=test_size)
        pickle.dump(train_normalized_features_table, open(variables["backup_folder"] + "train_normalized_features_table.p", "wb"))
        pickle.dump(test_normalized_features_table, open(variables["backup_folder"] + "test_normalized_features_table.p", "wb"))

        return train_normalized_features_table, test_normalized_features_table

    def generateRules(self, fuzzifier, normalized_features_table, variables, d_results, sigma_mean_params = -1, fuzzifyFive = False):
        features_table = normalized_features_table.copy()
        
        if fuzzifyFive:
            fuzzified_features_table, feature_labels, features, decision, fuzzify_parameters = fuzzifier.fuzzifyFive(features_table, sigma_mean_params)
        else:
            fuzzified_features_table, feature_labels, features, decision, fuzzify_parameters = fuzzifier.fuzzify(features_table, sigma_mean_params)
        pickle.dump(fuzzify_parameters, open(variables["backup_folder"] + "fuzzify_parameters.p", "wb"))
        pickle.dump(fuzzified_features_table, open(variables["backup_folder"] + "fuzzified_features_table.p", "wb"))
        pickle.dump(feature_labels, open(variables["backup_folder"] + "feature_labels.p", "wb"))
        pickle.dump(features, open(variables["backup_folder"] + "features.p", "wb"))
        
        inconsistencies_remover = InconsistenciesRemover(fuzzified_features_table, feature_labels, variables)
        decision_table = inconsistencies_remover.inconsistenciesRemoving()
        pickle.dump(decision_table, open(variables["backup_folder"] + "decision_table.p", "wb"))
        
        reductor = Reductor(decision_table, variables)
        decision_table_with_reduct = reductor.worker(decision_table)
        pickle.dump(decision_table_with_reduct, open(variables["backup_folder"] + "decision_table_with_reduct.p", "wb"))

        rules_extractor = RulesExtractor(decision_table_with_reduct, reductor.reduct, variables)
        rules, feature_names = rules_extractor.worker(decision_table_with_reduct, features, d_results, decision)
        pickle.dump(rules, open(variables["backup_folder"] + "rules.p", "wb"))
        pickle.dump(rules_extractor, open(variables["backup_folder"] + "rules_extractor.p", "wb"))
        
        if rules is None:
            return None, feature_names, features_table 
        else:
            return rules, feature_names, features_table
        
    def setRules(self, rules, variables, features_table):
        rulesSetter = RulesSetter()
        sorted_decision = rulesSetter.setRules(rules, features_table)
        pickle.dump(sorted_decision, open(variables["backup_folder"] + "sorted_decision.p", "wb"))
        
        return sorted_decision

    def workerCSV(self, variables, sigma_mean_params = -1):
        variables, d_results, levels = self.saveVariables(variables)
        fuzzifier, normalized_features_table = self.prepareData(variables, levels, d_results)
        normalized_features_table, test_normalized_features_table = self.splitDataForTrainingTest(normalized_features_table, variables)
        rules, feature_names, features_table = self.generateRules(fuzzifier, normalized_features_table, variables, d_results, sigma_mean_params, variables['fuzzify_five'])
        sorted_decision = self.setRules(rules, variables, normalized_features_table)

        print("Finished!")