import pandas as pd
from tqdm import tqdm
class InconsistenciesRemover(object):

    def __init__(self, features_table, feature_labels, variables):
        self.features_table = features_table
        self.feature_labels = feature_labels
        self.variables = variables

    def getOccurenceOfRows(self, df, remove_columns):
        if remove_columns:
            df = df.drop(remove_columns, axis=1, inplace=False)
        df = df.groupby(
            df.columns.tolist(),
            as_index=False).size().reset_index(name="Occurence")
        return df

    def getOccurenceOfRowsWithoutRemove(self, df):
        df = df.groupby(
            df.columns.tolist(),
            as_index=False).size().reset_index(name="Occurence")
        return df


    def getCertainDecisionRows(self, features_occurence, features_decisions_occurence):
        features_decision_numbers_ones = features_occurence.loc[
            features_occurence['Occurence'] == 1].copy()

        for index, row in features_decision_numbers_ones.iterrows():
            for idx, row_with_decision in features_decisions_occurence.iterrows():
                if (row[self.feature_labels].values ==
                        row_with_decision[self.feature_labels].values).all():
                    features_decision_numbers_ones.at[
                        index, 'Decision'] = features_decisions_occurence.loc[
                            idx, 'Decision']

        return features_decision_numbers_ones.drop(['Occurence'],
                                                axis=1,
                                                inplace=False)


    def getNumberOfClearDecision(self, features_occurence, features_decisions_occurence):
        features_certain_decision = self.getCertainDecisionRows(
            features_occurence, features_decisions_occurence)

        tmp_table = pd.merge(
            features_decisions_occurence,
            features_certain_decision,
            how='inner',
            on=self.feature_labels)

        if 'Decision_y' in tmp_table.columns:
            tmp_table = tmp_table.drop(['Decision_y'], axis=1).rename(
                index=str,
                columns={
                    "Decision_x": "Decision",
                    "Occurence_x": "Occurence"
                })

        number_of_clear_decision = pd.DataFrame(
            tmp_table.groupby(['Decision'],
                            as_index=False)['Occurence'].agg('sum'))

        return number_of_clear_decision


    def solveConflicts(self, number_of_conflicts_decision, problems_to_solve,
                        features_decisions_occurence, number_of_clear_decision):

        for _, row in number_of_conflicts_decision.iterrows():
            new_df = pd.DataFrame(columns={"Decision", "Probability"})

            for _, row_2 in problems_to_solve.iterrows():
                if (row[self.feature_labels].values == row_2[self.feature_labels]).all():

                    try:
                        occurence = (number_of_clear_decision.loc[
                            number_of_clear_decision['Decision'] == row_2[
                                ['Decision']].values[0]]).values[0][1]
                    except:
                        occurence = 0

                    probability = occurence / len(self.features_table)
                    new_df = new_df.append({
                        'Decision': row_2[['Decision']].values,
                        'Probability': probability
                    },
                                        ignore_index=True)

            new_value = new_df.loc[new_df['Probability'].idxmax()]['Decision'][0]

            for idx, row_decision_table in features_decisions_occurence.iterrows():
                if (row[self.feature_labels].values ==
                        row_decision_table[self.feature_labels]).all():
                    features_decisions_occurence.loc[idx, 'Decision'] = new_value

        return features_decisions_occurence

    def inconsistenciesRemoving(self):

        features_decisions_occurence = self.getOccurenceOfRows(
            self.features_table, ['Image'])
        
        if self.variables["show_results"]:
            display(features_decisions_occurence)


        features_occurence = self.getOccurenceOfRows(self.features_table,
                                                ['Image', 'Decision'])

        if self.variables["show_results"]:
            display(features_decisions_occurence)

        features_occurence = self.getOccurenceOfRows(self.features_table,
                                                ['Decision'])
        if self.variables["show_results"]:
            display(features_occurence)

        number_of_conflicts_decision = features_occurence[
            features_occurence.Occurence > 1]
        if self.variables["show_results"]:
            print("\nW tylu konflikach występuje:")
            display(number_of_conflicts_decision)

        number_of_clear_decision = self.getNumberOfClearDecision(
            features_occurence, features_decisions_occurence)
        if self.variables["show_results"]:
            print("\nTyle jest wystąpień takich czystych decyzji:")
            display(number_of_clear_decision)

        problems_to_solve = pd.merge(
            features_decisions_occurence,
            number_of_conflicts_decision,
            how='inner',
            on=self.feature_labels).drop(['Occurence_x', "Occurence_y"], axis=1)
        
        if self.variables["show_results"]:
            print("\nTe problemy należy rozwiązać:")
            display(problems_to_solve)

        features_decisions_occurence = self.solveConflicts(
            number_of_conflicts_decision, problems_to_solve,
            features_decisions_occurence, number_of_clear_decision)
        decision_table = features_decisions_occurence.drop(['Occurence'],
                                                        axis=1).drop_duplicates(
                                                            keep='first',
                                                            inplace=False)
        if self.variables["show_results"]:
            print("Tablica decyzyjna po usunięciu duplikatów i niespójności")
            display(decision_table)

        return decision_table