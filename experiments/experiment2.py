import matplotlib.pyplot as plt
import numpy as np

from usa.reader import ReadFromCSV
from usa.classifier import SimpleClassifier,\
    ClosureClassifier,\
    HypothesisClassifier
from usa.metrics import accuracy_score_with_unclassified_objects, tpr_fpr_nonclass

# model.fit(women_train_data,
#           men_train_data,
#           threshold_for_rules=0.003,
#           threshold_for_growth_rate=0.3)

file_name = '/Users/danil.gizdatullin/git_projects/HSE/UnbrokenSequenceAnalysis/examples/data/full_data_shuffle.csv'
coding_dict={'work': 1, 'separation': 2, 'partner': 3,
             'marriage': 4, 'children': 5, 'parting': 6,
             'divorce': 7, 'education': 8}
reader = ReadFromCSV(file_name, coding_dict)

data, label = reader.from_file_to_data_list()

size_of_train = int(len(data)*0.66)

X_train = data[:size_of_train]
X_test = data[size_of_train:]

y_train = label[:size_of_train]
y_test = label[size_of_train:]

origin_classifier = SimpleClassifier(number_of_classes=2,
                                     threshold_for_rules=0.003,
                                     threshold_for_growth_rate=0.3)

closure_classifier = ClosureClassifier(number_of_classes=2,
                                       threshold_for_rules=0.003,
                                       threshold_for_growth_rate=0.3)

hypothesis_classifier = HypothesisClassifier(number_of_classes=2,
                                             threshold_for_rules=0.0)

origin_classifier.fit(X_train, y_train)
closure_classifier.fit(X_train, y_train)
hypothesis_classifier.fit(X_train, y_train)

y_pred_origin = origin_classifier.predict(X_test)
y_pred_closure = closure_classifier.predict(X_test)
y_pred_hypothesis = hypothesis_classifier.predict(X_test)

print(len(y_test))

print(accuracy_score_with_unclassified_objects(y_test, y_pred_origin))
print(accuracy_score_with_unclassified_objects(y_test, y_pred_closure))
print(accuracy_score_with_unclassified_objects(y_test, y_pred_hypothesis))

print(tpr_fpr_nonclass(y_test, y_pred_origin))
print(tpr_fpr_nonclass(y_test, y_pred_closure))
print(tpr_fpr_nonclass(y_test, y_pred_hypothesis))

tp_o, fp_o, m_o = tpr_fpr_nonclass(y_test, y_pred_origin)
tp_c, fp_c, m_c = tpr_fpr_nonclass(y_test, y_pred_closure)
tp_h, fp_h, m_h = tpr_fpr_nonclass(y_test, y_pred_hypothesis)

y = [tp_o, tp_c, tp_h]
x = [fp_o, fp_c, fp_h]

plt.scatter(x, y)

X_plot = np.linspace(0, 1, 100)
plt.plot(X_plot, X_plot)
plt.axis([0, 1, 0, 1])
plt.show()
