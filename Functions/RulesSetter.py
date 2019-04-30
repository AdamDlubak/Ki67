from skfuzzy import control as ctrl

class RulesSetter(object):

    def setRules(self, rules, features_table):
        tipping_ctrl = ctrl.ControlSystem(rules)
        gen = tipping_ctrl.fuzzy_variables

        rules_feature_names = []
        for x in gen:
            if str(x).startswith('Antecedent'):
                rules_feature_names.append(str(x).split(': ')[1])
        
        tipping = ctrl.ControlSystemSimulation(tipping_ctrl)
        test_features_table = features_table[rules_feature_names].copy()
        test_features_table['Decision'] = features_table.Decision
        test_features_table['Decision Fuzzy'] = ""

        return test_features_table, rules_feature_names, tipping
