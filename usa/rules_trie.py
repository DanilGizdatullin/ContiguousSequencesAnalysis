import numpy as np

INF_VALUE = np.inf


class RulesTrie:
    def __init__(self, list_of_sequences=[], label=[]):
        """
        Initialize Trie for list of sequences.

        nodes - set -the set of tree nodes.
        node_childes_dict - dict - key is a node_id, and value is a list of children [node_id_1, ... , node_id_n].
        node_sequence_dict - dict - key is a node_id, value is a list of sequence for current node.
        node_visits_dict - dict - key is a node_id, value is a list^ where of i-th place the number of times objects
        from class i had this sequence.
        node_parent_dict - dict - key is node_id, value is a parrent node_id.
        node_full_sequence_dict - dict - key is node_id, value list of sequence from root to current node.

        :param list_of_sequences: list of lists - some representation of sequences
        :param label: - list of int from 0 to n where n-number of different classes
        :return:
        """
        self.nodes = set([])
        self.nodes_label = {}
        self.node_children_dict = {}
        self.node_parent_dict = {}
        self.node_sequence_dict = {}
        self.node_visits_dict = {}
        self.node_full_sequence_dict = {}
        self.list_of_sequences = list_of_sequences
        self.number_of_objects = len(list_of_sequences)
        self.number_of_classes = len(set(label))
        self.label = label

        nodes = set([])
        structure = {}
        dict_seq = {}
        dict_num = {}
        dict_prev = {}
        dict_all_seq = {}

        num_of_sequence = 0
        # len_of_seq = len(self.list_of_sequences)
        free_node = 1
        seq1 = self.list_of_sequences[0]
        nodes.add(0)
        current_node = 0
        dict_all_seq[0] = []

        for i in seq1:
            structure[current_node] = [free_node]
            structure[free_node] = []
            dict_prev[free_node] = current_node

            str_seq = [elem for elem in i]
            dict_all_seq[free_node] = dict_all_seq[current_node][:]
            dict_all_seq[free_node].append(str_seq)

            if type(str_seq) == list:
                dict_seq[free_node] = str_seq
            else:
                dict_seq[free_node] = [str_seq]
            # dict_num[free_node] = 1
            dict_num[free_node] = [0 for i in xrange(self.number_of_classes)]
            dict_num[free_node][self.label[num_of_sequence]] = 1
            current_node = free_node
            free_node += 1

        for seq in self.list_of_sequences[1:]:
            num_of_sequence += 1
            current_node = 0
            for elem in seq:
                str_seq = [i for i in elem]
                if len(structure[current_node]) > 0:
                    temp_seq = [dict_seq[son] for son in structure[current_node]]
                    flag = str_seq in temp_seq
                    if flag:
                        number = 0
                        while temp_seq[number] != str_seq:
                            number += 1
                        current_node = structure[current_node][number]
                        # dict_num[current_node] += 1
                        dict_num[current_node][self.label[num_of_sequence]] += 1
                    else:
                        structure[current_node].append(free_node)
                        dict_prev[free_node] = current_node
                        dict_all_seq[free_node] = dict_all_seq[current_node][:]
                        dict_all_seq[free_node].append(str_seq)
                        current_node = free_node
                        structure[current_node] = []
                        # dict_num[current_node] = 1
                        dict_num[current_node] = [0 for i in xrange(self.number_of_classes)]
                        dict_num[current_node][self.label[num_of_sequence]] = 1
                        dict_seq[current_node] = str_seq
                        free_node += 1
                else:
                    structure[current_node].append(free_node)
                    dict_prev[free_node] = current_node
                    dict_all_seq[free_node] = dict_all_seq[current_node][:]
                    dict_all_seq[free_node].append(str_seq)
                    current_node = free_node
                    structure[current_node] = []
                    # dict_num[current_node] = 1
                    dict_num[current_node] = [0 for i in xrange(self.number_of_classes)]
                    dict_num[current_node][self.label[num_of_sequence]] = 1
                    dict_seq[current_node] = str_seq
                    free_node += 1

        self.nodes = nodes
        self.node_children_dict = structure
        self.node_sequence_dict = dict_seq
        self.node_visits_dict = dict_num
        self.node_parent_dict = dict_prev
        self.node_full_sequence_dict = dict_all_seq

        dic_all_seq_rev = {str(v): k for k, v in self.node_full_sequence_dict.iteritems()}
        self.node_full_sequence_dict_reversed = dic_all_seq_rev

    def support_t(self, rule, label=0):
        """
        This function computes support for rule for current class

        :param rule: list, represents some sequence
        :param label: int, it's a class
        :return: float value from 0 to 1
        """
        # dic_all_seq_rev = {str(v): k for k, v in self.node_full_sequence_dict.iteritems()}
        dic_all_seq_rev = self.node_full_sequence_dict_reversed

        try:
            node = dic_all_seq_rev[str(rule)]
        except KeyError:
            node = -1
        if node == -1:
            sup = 0
        else:
            sup = self.node_visits_dict[node][label]

        number_of_objects = 0
        for i in self.label:
            if i == label:
                number_of_objects += 1
        return sup/float(number_of_objects)

    def support_t_except_class(self, rule, label=0):
        """
        This function computes supports for rule for all posible classes except lable class, and return maximum one.

        :param rule: list, represents some sequence
        :param label: int, it's a class
        :return: float value from 0 to 1
        """
        all_classes = [i for i in xrange(self.number_of_classes)]
        all_classes.remove(label)
        max_sup = self.support_t(rule, label=all_classes[0])

        for i in xrange(1, len(all_classes)):
            sup = self.support_t(rule, label=all_classes[i])
            if sup > max_sup:
                max_sup = sup

        return max_sup

    def important_rules_selection(self, min_threshold, label=0):
        """
        This function find rules which have support bigger than some threshold

        :param min_threshold: float from 0 to 1
        :param label: int value from 0 to n, where n is a number of different classes
        :return: list of "important" rules
        """
        ds_rules = []
        for item in self.node_full_sequence_dict.items():
            rule = item[1]
            if rule:
                if self.support_t(rule, label) > min_threshold:
                    ds_rules.append(rule)

        return ds_rules


class ClosureRulesTrie(RulesTrie):
    def is_closure(self, rule, label=0):
        """
        This boolean function return True if rule is closure and False otherwise

        :param rule: list of some rule
        :param label: int value for class
        :return: bool
        """
        node_id = self.node_full_sequence_dict_reversed[str(rule)]
        sup_node = self.node_visits_dict[node_id][label]

        children_sup = []
        for i in self.node_children_dict[node_id]:
            children_sup.append(self.node_visits_dict[i][label])
        try:
            max_child_sup = max(children_sup)
        except ValueError:
            max_child_sup = sup_node - 1

        if max_child_sup > sup_node:
            print "What the fuck?"
            return False
        elif max_child_sup == sup_node:
            return False
        else:
            return True

    def important_rules_selection(self, min_threshold, label=0):
        """
        This function find rules which have support bigger than some threshold

        :param min_threshold: float from 0 to 1
        :param label: int value from 0 to n, where n is a number of different classes
        :return: list of "important" rules
        """
        ds_rules = []
        for item in self.node_full_sequence_dict.items():
            rule = item[1]
            if rule:
                if self.support_t(rule, label) > min_threshold and self.is_closure(rule, label):
                    ds_rules.append(rule)

        return ds_rules

    # def find_second_node(self, part_of_rule, currend_node_id):
    #     """
    #     This method choose one of the children nodes which has part of rule, or return 0
    #
    #     :param part_of_rule: list of events or list of one event
    #     :param currend_node_id: int node_id
    #     :return: int node_id
    #     """
    #     new_node = 0
    #     for i in self.node_children_dict[currend_node_id]:
    #         if self.node_sequence_dict[i] == part_of_rule:
    #             new_node = i
    #
    #     return new_node

    # def support_t(self, rule, label=0):
    #     """
    #     This function computes support for rule for current class
    #
    #     :param rule: list, represents some sequence
    #     :param label: int, it's a class
    #     :return: float value from 0 to 1
    #     """
    #
    #     start_node = 0
    #     sup = 0
    #
    #     for i in self.node_children_dict[0]:
    #         if self.node_sequence_dict[i] == rule[0]:
    #             start_node = i
    #
    #     if start_node == 0:
    #         sup = 0
    #     else:
    #         new_node = start_node
    #         current_node = start_node
    #         for i in xrange(1, len(rule)):
    #             new_node = self.find_second_node(rule[i], current_node)
    #             if new_node == 0:
    #                 break
    #             else:
    #                 current_node = new_node
    #
    #         if new_node == 0:
    #             sup = 0
    #         else:
    #             sup = self.node_visits_dict[new_node][label]
    #
    #     number_of_objects = 0
    #     for i in self.label:
    #         if i == label:
    #             number_of_objects += 1
    #
    #     return sup/float(number_of_objects)


# class HypothesisRulesTrie(ClosureRulesTrie):
#     def hypothesis_selection(self, label=0):
#         """
#         This function find only hypothesis rules
#
#         :param label: int value from 0 to n, where n is a number of different classes
#         :return: list of "important" rules
#         """
#         min_threshold = np.inf
#
#         ds_rules = []
#         for item in self.node_full_sequence_dict.items():
#             rule = item[1]
#             if rule:
#                 if self.support_t(rule, label) > min_threshold and self.is_closure(rule, label):
#                     ds_rules.append(rule)
#
#         return ds_rules


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


class HypothesisImportance:
    def __init__(self, rules, trie, label):
        """
        This function creates a structure with rules importance for classification. This structure is two dictionaries.
        One has key as rule_id

        :param rules: list of sequences(rules)
        :param trie: some RulesTrie based on some data set
        :param threshold: threshold on growth rate to put rules in classifier
        :param label: class for what we want to make a set of important rules
        :return:
        """

        threshold = np.inf

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

if __name__ == '__main__':
    tree = RulesTrie(list_of_sequences=[[[1], [2], [3]], [[1], [2], [4]], [[1, 2], [3, 4], [5]], [[1], [3], [2]]],
                     label=[1, 1, 0, 1])
    # tree.trie_for_rules()
    print tree.nodes
    print tree.node_childes_dict
    print tree.node_parent_dict
    print tree.node_sequence_dict
    print tree.node_visits_dict
    print tree.node_full_sequence_dict
    print tree.list_of_sequences
    print tree.number_of_objects
    print(' ')
    print(tree.important_rules_selection(0.5, 1))
    print(tree.important_rules_selection(0.5, 0))
