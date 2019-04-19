import numpy as np
import boolean
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class RulesExtractor(object):

    def __init__(self, features_table, reduct, variables):
        self.features_table = features_table
        self.reduct = reduct
        self.variables = variables
        self.algebra = boolean.BooleanAlgebra()
        self.TRUE, self.FALSE, self.NOT, self.AND, self.OR, self.Symbol = self.algebra.definition()

    def generateSymetricDiscernibilityMatrix(self):

        self.features_table = self.features_table.drop(['Decision'], axis=1)

        symetric_discernibility_matrix = np.zeros((len(self.features_table),
                                                len(self.features_table),
                                                len(self.reduct)))

        for index_i, i in self.features_table.iterrows():
            for index_j, j in self.features_table.iterrows():
                if index_i != index_j:  # Tutaj jest różnica - nie ma ">" jest za to != tak aby uzyskać symetryczną tablicę
                    tmp_entity = np.zeros(len(self.reduct))
                    for idx, x in enumerate(range(0, len(i.values))):
                        if i[x] != j[x]:
                            tmp_entity[idx] = True
                    symetric_discernibility_matrix[int(index_i),
                                                int(index_j)] = tmp_entity

        if self.features_table.empty:
            return self.features_table

        return symetric_discernibility_matrix, self.features_table

    def generateImplicantsMatrix(self, symetric_discernibility_matrix, features_table):
        M_matrix = np.zeros((len(symetric_discernibility_matrix),
                            len(symetric_discernibility_matrix), len(self.reduct)))
        for index_i, i in features_table.iterrows():
            for index_j, j in features_table.iterrows():
                if i['Decision'] != j['Decision']:
                    M_matrix[index_i][index_j] = symetric_discernibility_matrix[
                        index_i][index_j]
        return M_matrix, features_table

    def getFeatureNames(self, features_table):
        feature_names = np.asarray(features_table.columns)[0:-1]

        if self.variables["show_results"]:
            print(feature_names)

        return feature_names


    def generateImplicants(self, implicants_matrix, feature_names):
        implicants = []
        for index_i, i in enumerate(implicants_matrix):
            new_expression = ""
            for index_j, j in enumerate(i):
                new_subexpression = "("
                any_value = False
                for index_k, k in enumerate(j):
                    if k == True:
                        any_value = True
                        new_subexpression = new_subexpression + feature_names[
                            index_k] + " | "
                if any_value:
                    new_subexpression = new_subexpression[:-3] + ")"
                    if new_expression != "":
                        new_expression = new_expression + " & " + new_subexpression
                    else:
                        new_expression = new_subexpression

            if new_expression is "(" or not new_expression:
                # print("Expression empty")
                simplified_implicant = ""
            else:
                simplified_implicant = self.algebra.parse(new_expression).simplify()
            implicants.append(simplified_implicant)

        if self.variables["show_results"]:
                print(implicants)

        return implicants


    def getIndexOfRulesTable(self, features_table):

        rules_table = []

        for x in features_table['Decision'].drop_duplicates():
            rules_table.append(
                features_table.loc[features_table['Decision'] == x].index.tolist())

        if self.variables["show_results"]:
            print("Connection between implicants and rules:" + str(rules_table))

        return rules_table


    def modifyImplicants(self, elem, implicant_idx, features_table, features):
        if type(elem) == self.AND:

            for idx_x, x in enumerate(elem.args):
                if idx_x == 0:
                    small_rule = self.modifyImplicants(x, implicant_idx,
                                                features_table, features)
                else:
                    small_rule = small_rule & self.modifyImplicants(
                        x, implicant_idx, features_table, features)

        if type(elem) == self.OR:

            for idx_x, x in enumerate(elem.args):
                if idx_x == 0:
                    small_rule = self.modifyImplicants(x, implicant_idx,
                                                features_table, features)
                else:
                    small_rule = small_rule | self.modifyImplicants(
                        x, implicant_idx, features_table, features)

        if type(elem) == self.Symbol:
            implicant_column_name = str(elem)
            implicant_column_number = str(elem)[1:]
            antecedent_part = features[int(implicant_column_number)][
                features_table.loc[implicant_idx][implicant_column_name]]
            return antecedent_part

        if type(elem) == str:
            return ""

        return small_rule


    def modifyImplicantsForRules(self, implicants, features_table, features):

        new_implicants = []

        for idx_i, i in enumerate(implicants):
            new_implicants.append(self.modifyImplicants(i, idx_i, features_table, features))

        if self.variables["show_results"]:
            print(new_implicants)

        return new_implicants


    def generateRuleAntecedents(self, index_of_rules_table, new_implicants):

        rule_antecedents = []
        for x in index_of_rules_table:
            for idx, y in enumerate(x):
                tmp_result = new_implicants[y]
                if not tmp_result:
                    # print("Empty implicant")
                    tmp_rule = ""
                else:
                    if idx == 0:
                        tmp_rule = new_implicants[y]
                    else:
                        tmp_rule = tmp_rule | new_implicants[y]
            if tmp_rule:
                rule_antecedents.append(tmp_rule)

        return rule_antecedents


    def generateRules(self, rule_antecedents, d_results, decision):
        rules = []
        for idx, x in enumerate(rule_antecedents):
            if x:
                rules.append(ctrl.Rule(x, decision[d_results[idx]]))

        return rules


    def worker(self, decision_table, features, d_results, decision):
        symetric_discernibility_matrix, features_table = self.generateSymetricDiscernibilityMatrix()
        if np.size(symetric_discernibility_matrix, 0) == 1:
            return None
        implicant_matrix, features_table = self.generateImplicantsMatrix(symetric_discernibility_matrix, decision_table)
        feature_names = self.getFeatureNames(features_table)
        implicants = self.generateImplicants(implicant_matrix, feature_names)
        index_of_rules_table = self.getIndexOfRulesTable(decision_table)
        rule_implicants = self.modifyImplicantsForRules(implicants, decision_table, features)
        rule_antecedents = self.generateRuleAntecedents(index_of_rules_table, rule_implicants) 
        rules = self.generateRules(rule_antecedents, d_results, decision)

        return rules, feature_names
    