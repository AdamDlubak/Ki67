import numpy as np

class Reductor(object):
    def __init__(self, decision_table, variables):
        self.decision_table = decision_table
        self.variables = variables
        
    def getReduct(self):
        self.reduct = []

        self.decision_table = self.decision_table.drop(['Decision'], axis=1)
        self.decision_table = self.decision_table.reset_index(drop=True)
        results = np.zeros((len(self.decision_table), len(self.decision_table),
                            len(self.decision_table.columns)))

        for index_i, i in self.decision_table.iterrows():
            for index_j, j in self.decision_table.iterrows():
                if index_i > index_j:
                    tmp_entity = np.zeros(len(self.decision_table.columns))
                    for idx, x in enumerate(range(0, len(i.values))):
                        if i[x] != j[x]:
                            tmp_entity[idx] = True
                    results[int(index_i), int(index_j)] = tmp_entity

        while True:
            reduct_counter = np.zeros(len(self.decision_table.columns))

            for i in results:
                for j in i:
                    for index_k, k in enumerate(j):
                        if k != 0:
                            reduct_counter[index_k] = reduct_counter[index_k] + 1

            if all(item == 0 for item in reduct_counter):
                break

            winner = np.argwhere(
                reduct_counter == np.amax(reduct_counter)).flatten().tolist()

            new_to_reduct = np.random.choice(winner)
            self.reduct.append(new_to_reduct)

            for index_i, i in enumerate(results):
                for index_j, j in enumerate(i):
                    if 1 == j[new_to_reduct]:
                        results[int(index_i)][int(index_j)] = np.zeros(
                            len(self.decision_table.columns))

            if all(item == 0 for item in reduct_counter):
                break

        if self.variables["show_results"]:
            print("Reduct - Index of features: " + str(self.reduct))



    def introduceReduct(self, decision_table):
        decision_table_after_reduct = decision_table[
            decision_table.columns[self.reduct]].copy()
        features_number_after_reduct = len(decision_table_after_reduct.columns.values)
        decision_table_after_reduct['Decision'] = decision_table['Decision'].copy()
        decision_table_after_reduct = decision_table_after_reduct.reset_index(drop=True)

        if self.variables["show_results"]:
            display(decision_table_after_reduct)

        return decision_table_after_reduct, features_number_after_reduct

    def worker(self, decision_table):
        self.getReduct()
        return self.introduceReduct(decision_table)