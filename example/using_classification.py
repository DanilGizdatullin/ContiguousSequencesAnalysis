from sklearn.metrics import accuracy_score

from usa.reader import read_from_csv
from usa.classifier import classifier_by_sequential_patterns as csp

sequence_reader = read_from_csv.SequencesFromFile(file_name='./data/full_data_shuffle.csv',
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

classifier = csp.Classifier(number_of_classes=2, threshold_for_rules=0.1, threshold_for_growth_rate=1.1)

classifier.fit(X_train, y_train)

test_pred = classifier.predict(X_test)

print(y_test)
print(test_pred)

print(accuracy_score(y_test, test_pred))
