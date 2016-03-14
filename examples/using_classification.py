from usa.reader import ReadFromCSV
from usa.metrics import accuracy_score_with_unclassified_objects, CostValueAbstainingClassifiers
from usa.classifier import ClassifierBySequencePatterns, ClassifierByClosureSequencePatterns,\
    ClassifierByHypothesisPatterns

sequence_reader = ReadFromCSV(file_name='./data/full_data_shuffle.csv',
                              coding_dict={'work': 1,
                                           'separation': 2,
                                           'partner': 3,
                                           'marriage': 4,
                                           'children': 5,
                                           'parting': 6,
                                           'divorce': 7,
                                           'education': 8})

data, label = sequence_reader.from_file_to_data_list(label_name='label')

size_of_train = int(len(data)*0.66)

X_train = data[:size_of_train]
X_test = data[size_of_train:]

y_train = label[:size_of_train]
y_test = label[size_of_train:]

# create object to calculate cost function for Abstaining Classifiers
costf = CostValueAbstainingClassifiers([1, 0.3])

# # create a classifier with support threshold and growth-rate threshold
# classifier = ClassifierBySequencePatterns(number_of_classes=2,
#                                           threshold_for_rules=0.01,
#                                           threshold_for_growth_rate=2.0)
# classifier.fit(X_train, y_train)
# test_pred = classifier.predict(X_test)
# print("Classification accuracy")
# print accuracy_score_with_unclassified_objects(y_test, test_pred)
# print("Cost function")
# print costf.expected_cost(y_test, test_pred)
# print("")
#
# # create classifier with Closure patterns
# classifier = ClassifierByClosureSequencePatterns(number_of_classes=2,
#                                                  threshold_for_rules=0.01,
#                                                  threshold_for_growth_rate=2.0)
# classifier.fit(X_train, y_train)
# test_pred = classifier.predict(X_test)
# print("Classification by closure patters accuracy")
# print accuracy_score_with_unclassified_objects(y_test, test_pred)
# print("Cost function")
# print costf.expected_cost(y_test, test_pred)
# print("")
#
# # create classifier with hypothesis
# classifier = ClassifierByHypothesisPatterns(number_of_classes=2,
#                                             threshold_for_rules=0.001,)
# classifier.fit(X_train, y_train)
# test_pred = classifier.predict(X_test)
# print("Classification by hypothesis accuracy")
# print accuracy_score_with_unclassified_objects(y_test, test_pred)
# print("Cost function")
# print costf.expected_cost(y_test, test_pred)
# print("")
classifier = ClassifierByClosureSequencePatterns(number_of_classes=2,
                                                 threshold_for_rules=0.001,
                                                 threshold_for_growth_rate=2.0)
classifier.fit(X_train, y_train)
test_pred = classifier.predict(X_test)
classifier.important_rules(0)
