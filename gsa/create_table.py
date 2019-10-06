import pandas as pd
import numpy as np
from reader import ReadFromCSV
from classifier import ClassifierByClosureSequencePatternsDifferentThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


class Table(object):
    def __init__(self, data, label, classifier=ClassifierByClosureSequencePatternsDifferentThreshold()):
        self.rules_class = classifier.rules_class
        # self.trie = classifier.trie
        self.data_frame = pd.DataFrame()

        full_stack_of_rules = []

        for i in range(2):
            dict_of_rules = self.rules_class[i].dict_of_rules
            rules = dict_of_rules.values()
            full_stack_of_rules += rules

        print(len(full_stack_of_rules))
        features = []
        for i in range(len(full_stack_of_rules)):
            features.append("f"+str(i))

        for rule_id in range(len(full_stack_of_rules)):
            feature_column = []
            for i in range(len(data)):
                if data[i][0: len(full_stack_of_rules[rule_id])] == full_stack_of_rules[rule_id]:
                    feature_column.append(1)
                else:
                    feature_column.append(0)
            self.data_frame[features[rule_id]] = feature_column

    def return_df(self):
        return self.final_data_frame


if __name__ == "__main__":
    file_name = '/Users/danil.gizdatullin/git_projects/HSE/UnbrokenSequenceAnalysis/examples/data/full_data_shuffle.csv'
    coding_dict = {'work': 1, 'separation': 2, 'partner': 3,
                 'marriage': 4, 'children': 5, 'parting': 6,
                 'divorce': 7, 'education': 8}

    reader = ReadFromCSV(file_name, coding_dict)

    data, label = reader.from_file_to_data_list()

    size_of_train = int(len(data)*0.66)

    X_train = data[:size_of_train]
    X_test = data[size_of_train:]

    y_train = label[:size_of_train]
    y_test = label[size_of_train:]

    classifier = ClassifierByClosureSequencePatternsDifferentThreshold(number_of_classes=2,
                                                                       threshold_for_rules0=0.04,
                                                                       threshold_for_growth_rate0=1.2,
                                                                       threshold_for_rules1=0.04,
                                                                       threshold_for_growth_rate1=1.2)
    classifier.fit(X_train, y_train)
    table = Table(X_train, y_train, classifier)
    X_train_df = table.data_frame
    X_train = X_train_df.values
    y_train = np.array(y_train)

    X_test = Table(X_test, y_test, classifier).data_frame
    y_test = np.array(y_test)

    model = DecisionTreeClassifier(class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(metrics.confusion_matrix(y_pred, y_test))