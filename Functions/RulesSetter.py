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

        for index, row in test_features_table.iterrows():
            new_dict = {}
            for x in rules_feature_names:
                new_dict[x] = row[x]

            tipping.inputs(new_dict)
            tipping.compute()
            test_features_table.loc[index, 'Predicted Value'] = tipping.output['Decision']

        sorted_decision = test_features_table.sort_values(by=['Predicted Value']).reset_index()

        return sorted_decision