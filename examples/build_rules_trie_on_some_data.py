from usa.reader import read_from_csv
from usa.rules_trie.trie import RulesTrie
from usa.rules_trie.important_rules import RulesImportance

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

rules_trie = RulesTrie(list_of_sequences=data, label=label)

print(rules_trie.node_full_sequence_dict)
print(rules_trie.support_t([['1']], 1))

rules_candidates = rules_trie.important_rules_selection(min_threshold=0.1, label=1)
print(rules_candidates)

important_rules = RulesImportance(rules_candidates, rules_trie, 0.1, 1)
print(important_rules.dict_of_contributions_to_score_class)
print(important_rules.dict_of_rules)
