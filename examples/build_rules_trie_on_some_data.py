from gsa.reader import ReadFromCSV
from gsa.rules_trie import RulesTrie, ClosureRulesTrie
from gsa.rules_trie import RulesImportance, HypothesisImportance

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
print(len(data))
# using data and label we can build Rules Trie and Closure Rules Trie
rules_trie = RulesTrie(list_of_sequences=data, label=label)
closure_rules_trie = ClosureRulesTrie(list_of_sequences=data, label=label)

# we can see for example the full trie structure and support for some sequence
print(rules_trie.node_full_sequence_dict)
print(rules_trie.support_t(rule=[['1']], label=1))
print(rules_trie.support_t(rule=[['1']], label=0))

print(closure_rules_trie.node_full_sequence_dict)
print(closure_rules_trie.support_t(rule=[['1']], label=1))
print(closure_rules_trie.support_t(rule=[['1']], label=0))
print("")

# also we can take important rules by some threshold
print("Rules with min support 0.2")
print(rules_trie.important_rules_selection(min_threshold=0.2, label=0))
print(rules_trie.important_rules_selection(min_threshold=0.2, label=1))
print("")
print(closure_rules_trie.important_rules_selection(min_threshold=0.2, label=0))
print(closure_rules_trie.important_rules_selection(min_threshold=0.2, label=1))


# also we can use a tool that uses in classification task, it takes rules with some growth rate threshold
# create some candidates by min support
rules_candidates_for1 = rules_trie.important_rules_selection(min_threshold=0.01, label=1)
rules_candidates_for0 = rules_trie.important_rules_selection(min_threshold=0.01, label=0)

# from candidates select important rules by threshold
important_rules_for0 = RulesImportance(rules=rules_candidates_for0, trie=rules_trie, threshold=2, label=0)
important_rules_for1 = RulesImportance(rules=rules_candidates_for1, trie=rules_trie, threshold=2, label=1)
print("Important Rules")
print(important_rules_for0.dict_of_rules)
print(important_rules_for1.dict_of_rules)
print("")

# the same actions but for closure patters
rules_candidates_for1 = closure_rules_trie.important_rules_selection(min_threshold=0.01, label=1)
rules_candidates_for0 = closure_rules_trie.important_rules_selection(min_threshold=0.01, label=0)
important_rules_for0 = RulesImportance(rules=rules_candidates_for0, trie=rules_trie, threshold=2, label=0)
important_rules_for1 = RulesImportance(rules=rules_candidates_for1, trie=rules_trie, threshold=2, label=1)
print("Important Closure Rules")
print(important_rules_for0.dict_of_rules)
print(important_rules_for1.dict_of_rules)
print("")

# the same actions but for hypothesis
rules_candidates_for1 = rules_trie.important_rules_selection(min_threshold=0.001, label=1)
rules_candidates_for0 = rules_trie.important_rules_selection(min_threshold=0.001, label=0)
important_rules_for0 = HypothesisImportance(rules=rules_candidates_for0, trie=rules_trie, label=0)
important_rules_for1 = HypothesisImportance(rules=rules_candidates_for1, trie=rules_trie, label=1)
print("Important Hypothesis")
print(important_rules_for0.dict_of_rules)
print(important_rules_for1.dict_of_rules)
print("")
