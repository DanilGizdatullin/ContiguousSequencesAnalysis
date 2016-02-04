import numpy as np

INF_VALUE = np.inf


def _growth_rate_t(rule, trie, label=0):
    """
    This function computes growth_rate(ratio of supports for different classes) for some rule in some trie-structure.

    :param rule: list, represents sequence
    :param trie: RulesTrie class structure
    :param label: int from 0 to n, where n-number of different classes
    :return: some float number of constant INF_VALUE
    """
    support_data_set0 = trie.support_t(rule, label)
    support_data_set1 = trie.support_t_except_class(rule, label)
    if (support_data_set0 == 0) & (support_data_set1 == 0):
        return 0
    elif (support_data_set0 != 0) & (support_data_set1 == 0):
        return INF_VALUE
    else:
        return support_data_set0 / float(support_data_set1)


class RulesImportance:
    def __init__(self, rules, trie, threshold, label):
        """
        This function creates a structure with rules importance for classification. This structure is two dictionaries.
        One has key as rule_id

        :param rules: list of sequences(rules)
        :param trie: some RulesTrie based on some data set
        :param threshold: threshold on growth rate to put rules in classifier
        :param label: class for what we want to make a set of important rules
        :return:
        """
        self.dict_of_contributions_to_score_class = {}
        self.dict_of_rules = {}

        for i in xrange(len(rules)):
            self.dict_of_rules[str(97 + i)] = rules[i]

        rules_to_delete = []

        for key in self.dict_of_rules.iterkeys():
            gr_ra1 = _growth_rate_t(self.dict_of_rules[key], trie, label)
            if gr_ra1 == INF_VALUE or gr_ra1 > threshold:
                if gr_ra1 == INF_VALUE:
                    self.dict_of_contributions_to_score_class[key] = trie.support_t(self.dict_of_rules[key], label)
                    self.dict_of_contributions_to_score_class[key] = INF_VALUE
                else:
                    # self.dict_of_contributions_to_score_class[key] = (gr_ra1 / (1 + gr_ra1)) * \
                    #                                                  (trie1.support_t(self.dict_of_rules[key]))
                    self.dict_of_contributions_to_score_class[key] = gr_ra1
            else:
                rules_to_delete.append(key)

        contributions = self.dict_of_contributions_to_score_class.values()
        contributions = np.array(contributions)
        median = np.median(contributions)
        # median = len(contributions)
        print("Median = %f" % median)

        # for key, value in self.dict_of_contributions_to_score_class.items():
        #     if self.dict_of_contributions_to_score_class[key] != INF_VALUE:
        #         self.dict_of_contributions_to_score_class[key] = value / float(median)

        for key in rules_to_delete:
            del self.dict_of_rules[key]
