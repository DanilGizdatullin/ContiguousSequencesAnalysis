import matplotlib.pyplot as plt
import numpy as np

from usa.reader import ReadFromCSV
from usa.classifier import ClassifierBySequencePatterns,\
    ClassifierByClosureSequencePatterns,\
    ClassifierByHypothesisPatterns
from usa.metrics import accuracy_score_with_unclassified_objects, tpr_fpr_nonclass

file_name = '/Users/danil.gizdatullin/git_projects/HSE/UnbrokenSequenceAnalysis/examples/data/full_data_shuffle.csv'
coding_dict={'work': 1, 'separation': 2, 'partner': 3,
             'marriage': 4, 'children': 5, 'parting': 6,
             'divorce': 7, 'education': 8}

reader = ReadFromCSV(file_name, coding_dict)

data, label = reader.from_file_to_data_list()

size_of_train = int(len(data)*0.66)

# print(size_of_train)

X_train = data[:size_of_train]
X_test = data[size_of_train:]

y_train = label[:size_of_train]
y_test = label[size_of_train:]

# origin_classifier = ClassifierBySequencePatterns(number_of_classes=2,
#                                                  threshold_for_rules=0.003,
#                                                  threshold_for_growth_rate=0.3)
f_ones = []
for i in (1.00001, 2, 5, 10, 20):
    origin_classifier = ClassifierBySequencePatterns(number_of_classes=2,
                                                     threshold_for_rules=0.0004,
                                                     threshold_for_growth_rate=i)
    origin_classifier.fit(X_train, y_train)
    y_pred = origin_classifier.predict(X_test, y_test)

