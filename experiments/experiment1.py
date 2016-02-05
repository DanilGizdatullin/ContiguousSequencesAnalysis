import numpy as np
import time
from sklearn.metrics import accuracy_score

from usa.reader import ReadFromCSV
from usa.classifier import ClassifierBySequencePatterns as csp
from usa.rules_trie import RulesTrie, ClosureRulesTrie, RulesImportance, HypothesisImportance


file_name = '/Users/danil.gizdatullin/git_projects/HSE/UnbrokenSequenceAnalysis/examples/data/full_data_shuffle.csv'
sequence_reader = ReadFromCSV(file_name=file_name,
                              coding_dict={'work': 1,
                                           'separation': 2,
                                           'partner': 3,
                                           'marriage': 4,
                                           'children': 5,
                                           'parting': 6,
                                           'divorce': 7,
                                           'education': 8})

data, label = sequence_reader.from_file_to_data_list(label_name='label')

print(data[235])
print(label[235])

len_of_sequences = []
for i in data:
    len_of_sequences.append(len(i))

print(np.argmax(len_of_sequences))
print(data[np.argmax(len_of_sequences)])

# size_of_train = int(len(data)*0.66)
#
# X_train = data[:size_of_train]
# X_test = data[size_of_train:]
#
# y_train = label[:size_of_train]
# y_test = label[size_of_train:]
#
# classifier = csp.Classifier(number_of_classes=2, threshold_for_rules=0.1, threshold_for_growth_rate=1.1)
#
# classifier.fit(X_train, y_train)
#
# test_pred = classifier.predict(X_test)
#
# print(y_test)
# print(test_pred)
#
# print(accuracy_score(y_test, test_pred))

trie = RulesTrie(data, label)
closure_trie = ClosureRulesTrie(data, label)

hypo_candidates = closure_trie.important_rules_selection(0.00002, 0)

rules = trie.important_rules_selection(0.001, 0)
closure_rules = closure_trie.important_rules_selection(0.001, 0)

all_hypothesis = HypothesisImportance(hypo_candidates, trie, 0)

# for i in rules:
#     flag = False
#     for j in closure_rules:
#         if i == j:
#             flag = True
#     if not flag:
#         print(i)

# print(trie.support_t([['8'], ['1']], 0))
# print(closure_trie.support_t([['8'], ['1']], 0))
# [['1']]
# [['1'], ['8']]
# [['2']], [['8']]
# [['8'], ['1']]
# [['2'], ['8']]


# not closure rules
# [['1'], ['8'], ['2'], ['4']]
# [['4', '2'], ['8', '5']]
# [['1', '8'], ['3'], ['4'], ['5']]
# [['4'], ['2'], ['5'], ['8']]
# [['1'], ['3'], ['4', '2']]
# [['1', '2', '8'], ['4']]
# [['1'], ['8'], ['3'], ['4', '2']]
# [['1'], ['2', '3'], ['8'], ['4']]
# [['2'], ['1'], ['3'], ['8'], ['4']]
# [['1'], ['4'], ['5'], ['8']]
# [['2', '8']]
# [['2', '8'], ['1'], ['4']]
# [['1'], ['2', '8'], ['4']]
# [['2'], ['8'], ['4'], ['1']]
# [['8'], ['1'], ['3'], ['4', '2']]
# [['1', '4', '2'], ['5']]
# [['2'], ['4', '8']]
# [['2'], ['4', '8'], ['1']]
# [['2'], ['1'], ['3'], ['4'], ['8']]
# [['8'], ['1'], ['3'], ['4'], ['5'], ['6'], ['7']]
# [['1', '8'], ['4'], ['2']]
# [['4'], ['5'], ['2'], ['8']]
# [['1', '8'], ['2', '3'], ['4']]
# [['1'], ['4'], ['8'], ['2']]
# [['1'], ['2'], ['3'], ['6']]
# [['1', '8'], ['2'], ['3'], ['4']]
# [['8'], ['4'], ['1', '2']]


# for rule in closure_rules:
#     if rule[0: 4] == [['1'], ['8'], ['2'], ['4']]:
#         print(rule)

# [['1'], ['8'], ['2'], ['4'], ['5']]
# [['1'], ['8'], ['2'], ['4'], ['5'], ['7']]

# print(closure_trie.support_t([['1'], ['8'], ['2'], ['4']], 0))
# print(closure_trie.support_t([['1'], ['8'], ['2'], ['4'], ['5']], 0))
# print(closure_trie.support_t([['1'], ['8'], ['2'], ['4'], ['5'], ['7']], 0))
#
# print(trie.support_t([['1'], ['8'], ['2'], ['4']], 0))
# print(trie.support_t([['1'], ['8'], ['2'], ['4'], ['5']], 0))
# print(trie.support_t([['1'], ['8'], ['2'], ['4'], ['5'], ['7']], 0))

# print(trie.support_t([['2'], ['1'], ['4'], ['5'], ['8'], ['7'], ['3'], ['6']], 0))
# print(closure_trie.support_t([['2'], ['1'], ['4'], ['5'], ['8'], ['7'], ['3'], ['6']], 0))
#
# print(trie.support_t([['2'], ['1'], ['4'], ['5'], ['8'], ['7'], ['3'], ['6']], 1))
# print(closure_trie.support_t([['2'], ['1'], ['4'], ['5'], ['8'], ['7'], ['3'], ['6']], 1))

print all_hypothesis.dict_of_rules
print all_hypothesis.dict_of_contributions_to_score_class
