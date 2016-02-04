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
        self.node_childes_dict = {}
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
        self.node_childes_dict = structure
        self.node_sequence_dict = dict_seq
        self.node_visits_dict = dict_num
        self.node_parent_dict = dict_prev
        self.node_full_sequence_dict = dict_all_seq

    def support_t(self, rule, label=0):
        """
        This function computes support for rule for current class

        :param rule: list, represents some sequence
        :param label: int, it's a class
        :return: float value from 0 to 1
        """
        dic_all_seq_rev = {str(v): k for k, v in self.node_full_sequence_dict.iteritems()}
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
