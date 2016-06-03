import numpy as np

from gsa.rules_trie import RulesImportance, HypothesisImportance
from gsa.rules_trie import RulesTrie, ClosureRulesTrie

INF_VALUE = np.inf


class ClassifierBySequencePatterns:
    def __init__(self, number_of_classes=2, threshold_for_rules=0.01, threshold_for_growth_rate=1.):
        self.model = False
        self.rules_class = []
        self.trie = False
        self.threshold_for_rules = threshold_for_rules
        self.threshold_for_growth_rate = threshold_for_growth_rate
        self.number_of_classes = number_of_classes

    def fit(self, data, label):

        # print(len(data))
        # print(len(label))

        rules_tree = RulesTrie(data, label)

        imp_rules = []
        for i in xrange(self.number_of_classes):
            imp_rules.append(rules_tree.important_rules_selection(self.threshold_for_rules, label=i))

        self.trie = rules_tree

        inf = []
        for i in xrange(self.number_of_classes):
            inf.append(RulesImportance(imp_rules[i], rules_tree, self.threshold_for_growth_rate, label=i))

        for i in inf:
            self.rules_class.append(i)

        self.model = True

    def predict(self, data_to_classification):
        result = []
        for obj in data_to_classification:
            result.append(self._classify_object(obj))

        return result

    def predict_proba(self, data_to_classification):
        result = []
        for obj in data_to_classification:
            result.append(self._classify_object_score(obj))

        return result

    def _rule_uncoding(self, rule=[], dict_of_uncoding={}):
        ans = []
        for part in rule:
            temp_part = []
            for elem in part:
                temp_part.append(dict_of_uncoding[elem])
            ans.append(temp_part)

        return ans

    def important_rules(self, label=0, dict_of_coding={'work': 1,
                                                       'separation': 2,
                                                       'partner': 3,
                                                       'marriage': 4,
                                                       'children': 5,
                                                       'parting': 6,
                                                       'divorce': 7,
                                                       'education': 8}):
        if self.model:
            rules = self.rules_class[label]
            dict_of_rules = rules.dict_of_rules
            dict_of_contributions = rules.dict_of_contributions_to_score_class

            dict_of_uncoding = {}
            for key, value in dict_of_coding.iteritems():
                dict_of_uncoding[str(value)] = key

            dict_of_stats = {}
            for key in dict_of_rules.iterkeys():
                dict_of_stats[key] = [dict_of_contributions[key]]
            for key, value in dict_of_rules.iteritems():
                dict_of_stats[key].append(self.trie.support_t(value, label))

            ans = []
            for key, value in dict_of_rules.iteritems():
                ans.append((self._rule_uncoding(value, dict_of_uncoding), dict_of_stats[key]))
            return ans
            # print dict_of_rules
            # print dict_of_contributions
        else:
            print("You need to train model!")

    def _classify_object(self, object_to_classification, silence=True):
        score_for_class = [0 for _ in xrange(self.number_of_classes)]

        rules_from_class = [[] for _ in xrange(self.number_of_classes)]

        for i in xrange(self.number_of_classes):
            for rule_id, rule in self.rules_class[i].dict_of_rules.items():
                rule_len = len(rule)
                if rule == object_to_classification[0: rule_len]:
                    rules_from_class[i].append(rule_id)

            id_to_remove_from_class = [[] for i in xrange(self.number_of_classes)]
            for rule_id1 in rules_from_class[i]:
                for rule_id2 in rules_from_class[i]:
                    if rule_id1 != rule_id2 and\
                            (rule_id1 not in id_to_remove_from_class[i]) and\
                            (rule_id2 not in id_to_remove_from_class[i]):
                        rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                        rule2 = self.rules_class[i].dict_of_rules[rule_id2]
                        if rule1[0: len(rule2)] == rule2:
                            id_to_remove_from_class[i].append(rule_id2)
                        elif rule2[0: len(rule1)] == rule1:
                            id_to_remove_from_class[i].append(rule_id1)
        for i in xrange(self.number_of_classes):
            for j in xrange(self.number_of_classes):
                for rule_id1 in rules_from_class[i]:
                    for rule_id2 in rules_from_class[j]:
                        if rule_id1 != rule_id2 and\
                                (rule_id1 not in id_to_remove_from_class[i]) and\
                                (rule_id2 not in id_to_remove_from_class[j]):
                            rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                            rule2 = self.rules_class[j].dict_of_rules[rule_id2]
                            if rule1[0: len(rule2)] == rule2:
                                id_to_remove_from_class[j].append(rule_id2)
                            elif rule2[0: len(rule1)] == rule1:
                                id_to_remove_from_class[i].append(rule_id1)

        for i in xrange(self.number_of_classes):
            # print(id_to_remove_from_class[i])
            for id_to_del in id_to_remove_from_class[i]:
                # print(rules_from_class[i], id_to_del)
                rules_from_class[i].remove(id_to_del)

        for i in xrange(self.number_of_classes):
            for rule in rules_from_class[i]:
                if rule in self.rules_class[i].dict_of_contributions_to_score_class:
                    score_for_class[i] += self.rules_class[i].dict_of_contributions_to_score_class[rule]
        if not silence:
            print(rules_from_class)
        if max(score_for_class) != 0:
            ans_class = np.argmax(score_for_class)

            return ans_class
        else:
            return -1

    def _classify_object_score(self, object_to_classification, silence=True):
        score_for_class = [0 for _ in xrange(self.number_of_classes)]

        rules_from_class = [[] for _ in xrange(self.number_of_classes)]

        for i in xrange(self.number_of_classes):
            for rule_id, rule in self.rules_class[i].dict_of_rules.items():
                rule_len = len(rule)
                if rule == object_to_classification[0: rule_len]:
                    rules_from_class[i].append(rule_id)

            id_to_remove_from_class = [[] for i in xrange(self.number_of_classes)]
            for rule_id1 in rules_from_class[i]:
                for rule_id2 in rules_from_class[i]:
                    if rule_id1 != rule_id2 and\
                            (rule_id1 not in id_to_remove_from_class[i]) and\
                            (rule_id2 not in id_to_remove_from_class[i]):
                        rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                        rule2 = self.rules_class[i].dict_of_rules[rule_id2]
                        # print(rule_id1)
                        # print(rule_id2)
                        # print(rule1)
                        # print(rule2)
                        if rule1[0: len(rule2)] == rule2:
                            id_to_remove_from_class[i].append(rule_id2)
                        elif rule2[0: len(rule1)] == rule1:
                            id_to_remove_from_class[i].append(rule_id1)

        for i in xrange(self.number_of_classes):
            for j in xrange(self.number_of_classes):
                for rule_id1 in rules_from_class[i]:
                    for rule_id2 in rules_from_class[j]:
                        if rule_id1 != rule_id2 and\
                                (rule_id1 not in id_to_remove_from_class[i]) and\
                                (rule_id2 not in id_to_remove_from_class[j]):
                            rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                            rule2 = self.rules_class[j].dict_of_rules[rule_id2]
                            if rule1[0: len(rule2)] == rule2:
                                id_to_remove_from_class[j].append(rule_id2)
                            elif rule2[0: len(rule1)] == rule1:
                                id_to_remove_from_class[i].append(rule_id1)

        for i in xrange(self.number_of_classes):
            # print(id_to_remove_from_class[i])
            for id_to_del in id_to_remove_from_class[i]:
                # print(rules_from_class[i], id_to_del)
                rules_from_class[i].remove(id_to_del)

        for i in xrange(self.number_of_classes):
            for rule in rules_from_class[i]:
                if rule in self.rules_class[i].dict_of_contributions_to_score_class:
                    score_for_class[i] += self.rules_class[i].dict_of_contributions_to_score_class[rule]

        if not silence:
            print(rules_from_class)

        return score_for_class

    # def important_rules(self, class_name=0):
    #     if self.model:
    #         rules = []
    #
    #         try:
    #             for rule_id in self.rules_class[class_name].dict_of_contributions_to_score_class.iterkeys():
    #                 rules.append((self.rules_class[class_name].dict_of_rules[rule_id],
    #                               self.rules_class[class_name].dict_of_contributions_to_score_class[rule_id]))
    #
    #             return rules
    #         except IndexError:
    #             print("Choose the right class name")
    #
    #             return []
    #     else:
    #         print("You need to fit model")
    #
    #         return []


class ClassifierByClosureSequencePatterns(ClassifierBySequencePatterns):
    def fit(self, data, label):
        rules_tree = ClosureRulesTrie(data, label)

        imp_rules = []
        for i in xrange(self.number_of_classes):
            imp_rules.append(rules_tree.important_rules_selection(self.threshold_for_rules, label=i))

        self.trie = rules_tree

        inf = []
        for i in xrange(self.number_of_classes):
            inf.append(RulesImportance(imp_rules[i], rules_tree, self.threshold_for_growth_rate, label=i))

        for i in inf:
            self.rules_class.append(i)

        self.model = True


class ClassifierByHypothesisPatterns(ClassifierBySequencePatterns):
    def fit(self, data, label):
        rules_tree = ClosureRulesTrie(data, label)

        imp_rules = []
        for i in xrange(self.number_of_classes):
            imp_rules.append(rules_tree.important_rules_selection(self.threshold_for_rules, label=i))

        self.trie = rules_tree

        inf = []
        for i in xrange(self.number_of_classes):
            inf.append(HypothesisImportance(imp_rules[i], rules_tree, label=i))

        for i in inf:
            self.rules_class.append(i)

        self.model = True


class ClassifierByClosureSequencePatternsWithWeighs(ClassifierByClosureSequencePatterns):
    def __init__(self, number_of_classes=2, threshold_for_rules=0.01, threshold_for_growth_rate=1., weights=(1., 1.)):
        self.model = False
        self.rules_class = []
        self.trie = False
        self.threshold_for_rules = threshold_for_rules
        self.threshold_for_growth_rate = threshold_for_growth_rate
        self.number_of_classes = number_of_classes
        self.weights = weights

    def _classify_object(self, object_to_classification, silence=True):
        score_for_class = [0 for _ in xrange(self.number_of_classes)]

        rules_from_class = [[] for _ in xrange(self.number_of_classes)]

        for i in xrange(self.number_of_classes):
            for rule_id, rule in self.rules_class[i].dict_of_rules.items():
                rule_len = len(rule)
                if rule == object_to_classification[0: rule_len]:
                    rules_from_class[i].append(rule_id)

            id_to_remove_from_class = [[] for i in xrange(self.number_of_classes)]
            for rule_id1 in rules_from_class[i]:
                for rule_id2 in rules_from_class[i]:
                    if rule_id1 != rule_id2 and\
                            (rule_id1 not in id_to_remove_from_class[i]) and\
                            (rule_id2 not in id_to_remove_from_class[i]):
                        rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                        rule2 = self.rules_class[i].dict_of_rules[rule_id2]
                        if rule1[0: len(rule2)] == rule2:
                            id_to_remove_from_class[i].append(rule_id2)
                        elif rule2[0: len(rule1)] == rule1:
                            id_to_remove_from_class[i].append(rule_id1)
        for i in xrange(self.number_of_classes):
            for j in xrange(self.number_of_classes):
                for rule_id1 in rules_from_class[i]:
                    for rule_id2 in rules_from_class[j]:
                        if rule_id1 != rule_id2 and\
                                (rule_id1 not in id_to_remove_from_class[i]) and\
                                (rule_id2 not in id_to_remove_from_class[j]):
                            rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                            rule2 = self.rules_class[j].dict_of_rules[rule_id2]
                            if rule1[0: len(rule2)] == rule2:
                                id_to_remove_from_class[j].append(rule_id2)
                            elif rule2[0: len(rule1)] == rule1:
                                id_to_remove_from_class[i].append(rule_id1)

        for i in xrange(self.number_of_classes):
            # print(id_to_remove_from_class[i])
            for id_to_del in id_to_remove_from_class[i]:
                # print(rules_from_class[i], id_to_del)
                rules_from_class[i].remove(id_to_del)

        for i in xrange(self.number_of_classes):
            for rule in rules_from_class[i]:
                if rule in self.rules_class[i].dict_of_contributions_to_score_class:
                    score_for_class[i] += self.rules_class[i].dict_of_contributions_to_score_class[rule]
        if not silence:
            print(rules_from_class)

        for i in xrange(len(score_for_class)):
            score_for_class[i] *= self.weights[i]

        if max(score_for_class) != 0:
            ans_class = np.argmax(score_for_class)

            return ans_class
        else:
            return -1


# class ClassifierWithDifferentThresholds(ClassifierBySequencePatterns):
#     def fit(self, data, label):
#         rules_tree = ClosureRulesTrie(data, label)
#
#         imp_rules = []
#         for i in xrange(self.number_of_classes):
#             imp_rules.append(rules_tree.important_rules_selection(self.threshold_for_rules, label=i))
#
#         self.trie = rules_tree
#
#         inf = []
#         for i in xrange(self.number_of_classes):
#             inf.append(HypothesisImportance(imp_rules[i], rules_tree, label=i))
#
#         for i in inf:
#             self.rules_class.append(i)
#
#         self.model = True

class ClassifierByClosureSequencePatternsDifferentThreshold(object):
    def __init__(self, number_of_classes=2, threshold_for_rules0=0.01, threshold_for_growth_rate0=1.,
                 threshold_for_rules1=0.01, threshold_for_growth_rate1=1.):
        self.model = False
        self.rules_class = []
        self.trie = False
        self.threshold_for_rules = [threshold_for_rules0, threshold_for_rules1]
        self.threshold_for_growth_rate = [threshold_for_growth_rate0, threshold_for_growth_rate1]
        self.number_of_classes = number_of_classes

    def fit(self, data, label):
        rules_tree = ClosureRulesTrie(data, label)

        imp_rules = []
        for i in xrange(self.number_of_classes):
            imp_rules.append(rules_tree.important_rules_selection(self.threshold_for_rules[i], label=i))

        self.trie = rules_tree

        inf = []
        for i in xrange(self.number_of_classes):
            inf.append(RulesImportance(imp_rules[i], rules_tree, self.threshold_for_growth_rate[i], label=i))

        for i in inf:
            self.rules_class.append(i)

        self.model = True

    def predict(self, data_to_classification):
        result = []
        for obj in data_to_classification:
            result.append(self._classify_object(obj))

        return result

    def predict_proba(self, data_to_classification):
        result = []
        for obj in data_to_classification:
            result.append(self._classify_object_score(obj))

        return result

    def _rule_uncoding(self, rule=[], dict_of_uncoding={}):
        ans = []
        for part in rule:
            temp_part = []
            for elem in part:
                temp_part.append(dict_of_uncoding[elem])
            ans.append(temp_part)

        return ans

    def important_rules(self, label=0, dict_of_coding={'work': 1,
                                                       'separation': 2,
                                                       'partner': 3,
                                                       'marriage': 4,
                                                       'children': 5,
                                                       'parting': 6,
                                                       'divorce': 7,
                                                       'education': 8}):
        if self.model:
            rules = self.rules_class[label]
            dict_of_rules = rules.dict_of_rules
            dict_of_contributions = rules.dict_of_contributions_to_score_class

            dict_of_uncoding = {}
            for key, value in dict_of_coding.iteritems():
                dict_of_uncoding[str(value)] = key

            dict_of_stats = {}
            for key in dict_of_rules.iterkeys():
                dict_of_stats[key] = [dict_of_contributions[key]]
            for key, value in dict_of_rules.iteritems():
                dict_of_stats[key].append(self.trie.support_t(value, label))

            ans = []
            for key, value in dict_of_rules.iteritems():
                ans.append((self._rule_uncoding(value, dict_of_uncoding), dict_of_stats[key]))
            return ans
            # print dict_of_rules
            # print dict_of_contributions
        else:
            print("You need to train model!")

    def _classify_object(self, object_to_classification, silence=True):
        score_for_class = [0 for _ in xrange(self.number_of_classes)]

        rules_from_class = [[] for _ in xrange(self.number_of_classes)]

        for i in xrange(self.number_of_classes):
            for rule_id, rule in self.rules_class[i].dict_of_rules.items():
                rule_len = len(rule)
                if rule == object_to_classification[0: rule_len]:
                    rules_from_class[i].append(rule_id)

            id_to_remove_from_class = [[] for i in xrange(self.number_of_classes)]
            for rule_id1 in rules_from_class[i]:
                for rule_id2 in rules_from_class[i]:
                    if rule_id1 != rule_id2 and\
                            (rule_id1 not in id_to_remove_from_class[i]) and\
                            (rule_id2 not in id_to_remove_from_class[i]):
                        rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                        rule2 = self.rules_class[i].dict_of_rules[rule_id2]
                        if rule1[0: len(rule2)] == rule2:
                            id_to_remove_from_class[i].append(rule_id2)
                        elif rule2[0: len(rule1)] == rule1:
                            id_to_remove_from_class[i].append(rule_id1)
        for i in xrange(self.number_of_classes):
            for j in xrange(self.number_of_classes):
                for rule_id1 in rules_from_class[i]:
                    for rule_id2 in rules_from_class[j]:
                        if rule_id1 != rule_id2 and\
                                (rule_id1 not in id_to_remove_from_class[i]) and\
                                (rule_id2 not in id_to_remove_from_class[j]):
                            rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                            rule2 = self.rules_class[j].dict_of_rules[rule_id2]
                            if rule1[0: len(rule2)] == rule2:
                                id_to_remove_from_class[j].append(rule_id2)
                            elif rule2[0: len(rule1)] == rule1:
                                id_to_remove_from_class[i].append(rule_id1)

        for i in xrange(self.number_of_classes):
            # print(id_to_remove_from_class[i])
            for id_to_del in id_to_remove_from_class[i]:
                # print(rules_from_class[i], id_to_del)
                rules_from_class[i].remove(id_to_del)

        for i in xrange(self.number_of_classes):
            for rule in rules_from_class[i]:
                if rule in self.rules_class[i].dict_of_contributions_to_score_class:
                    score_for_class[i] += self.rules_class[i].dict_of_contributions_to_score_class[rule]
        if not silence:
            print(rules_from_class)
        if max(score_for_class) != 0:
            ans_class = np.argmax(score_for_class)

            return ans_class
        else:
            return -1

    def _classify_object_score(self, object_to_classification, silence=True):
        score_for_class = [0 for _ in xrange(self.number_of_classes)]

        rules_from_class = [[] for _ in xrange(self.number_of_classes)]

        for i in xrange(self.number_of_classes):
            for rule_id, rule in self.rules_class[i].dict_of_rules.items():
                rule_len = len(rule)
                if rule == object_to_classification[0: rule_len]:
                    rules_from_class[i].append(rule_id)

            id_to_remove_from_class = [[] for i in xrange(self.number_of_classes)]
            for rule_id1 in rules_from_class[i]:
                for rule_id2 in rules_from_class[i]:
                    if rule_id1 != rule_id2 and\
                            (rule_id1 not in id_to_remove_from_class[i]) and\
                            (rule_id2 not in id_to_remove_from_class[i]):
                        rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                        rule2 = self.rules_class[i].dict_of_rules[rule_id2]
                        # print(rule_id1)
                        # print(rule_id2)
                        # print(rule1)
                        # print(rule2)
                        if rule1[0: len(rule2)] == rule2:
                            id_to_remove_from_class[i].append(rule_id2)
                        elif rule2[0: len(rule1)] == rule1:
                            id_to_remove_from_class[i].append(rule_id1)

        for i in xrange(self.number_of_classes):
            for j in xrange(self.number_of_classes):
                for rule_id1 in rules_from_class[i]:
                    for rule_id2 in rules_from_class[j]:
                        if rule_id1 != rule_id2 and\
                                (rule_id1 not in id_to_remove_from_class[i]) and\
                                (rule_id2 not in id_to_remove_from_class[j]):
                            rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                            rule2 = self.rules_class[j].dict_of_rules[rule_id2]
                            if rule1[0: len(rule2)] == rule2:
                                id_to_remove_from_class[j].append(rule_id2)
                            elif rule2[0: len(rule1)] == rule1:
                                id_to_remove_from_class[i].append(rule_id1)

        for i in xrange(self.number_of_classes):
            # print(id_to_remove_from_class[i])
            for id_to_del in id_to_remove_from_class[i]:
                # print(rules_from_class[i], id_to_del)
                rules_from_class[i].remove(id_to_del)

        for i in xrange(self.number_of_classes):
            for rule in rules_from_class[i]:
                if rule in self.rules_class[i].dict_of_contributions_to_score_class:
                    score_for_class[i] += self.rules_class[i].dict_of_contributions_to_score_class[rule]

        if not silence:
            print(rules_from_class)

        return score_for_class



# create a fast classification
# class EfficientClassifier(ClassifierByClosureSequencePatterns):
#     def _classify_object(self, object_to_classification, silence=True):
#         score_for_each_class = [0 for _ in xrange(self.number_of_classes)]
#         score_for_class = [0 for _ in xrange(self.number_of_classes)]
#
#         tree = self.trie
#
#         if max(score_for_class) != 0:
#             ans_class = np.argmax(score_for_class)
#
#             return ans_class
#         else:
#             return -1

class ClassifierSequencePatternsDifferentThreshold(object):
    def __init__(self, number_of_classes=2, threshold_for_rules0=0.01, threshold_for_growth_rate0=1.,
                 threshold_for_rules1=0.01, threshold_for_growth_rate1=1.):
        self.model = False
        self.rules_class = []
        self.trie = False
        self.threshold_for_rules = [threshold_for_rules0, threshold_for_rules1]
        self.threshold_for_growth_rate = [threshold_for_growth_rate0, threshold_for_growth_rate1]
        self.number_of_classes = number_of_classes

    def fit(self, data, label):
        rules_tree = RulesTrie(data, label)

        imp_rules = []
        for i in xrange(self.number_of_classes):
            imp_rules.append(rules_tree.important_rules_selection(self.threshold_for_rules[i], label=i))

        self.trie = rules_tree

        inf = []
        for i in xrange(self.number_of_classes):
            inf.append(RulesImportance(imp_rules[i], rules_tree, self.threshold_for_growth_rate[i], label=i))

        for i in inf:
            self.rules_class.append(i)

        self.model = True

    def predict(self, data_to_classification):
        result = []
        for obj in data_to_classification:
            result.append(self._classify_object(obj))

        return result

    def predict_proba(self, data_to_classification):
        result = []
        for obj in data_to_classification:
            result.append(self._classify_object_score(obj))

        return result

    def _rule_uncoding(self, rule=[], dict_of_uncoding={}):
        ans = []
        for part in rule:
            temp_part = []
            for elem in part:
                temp_part.append(dict_of_uncoding[elem])
            ans.append(temp_part)

        return ans

    def important_rules(self, label=0, dict_of_coding={'work': 1,
                                                       'separation': 2,
                                                       'partner': 3,
                                                       'marriage': 4,
                                                       'children': 5,
                                                       'parting': 6,
                                                       'divorce': 7,
                                                       'education': 8}):
        if self.model:
            rules = self.rules_class[label]
            dict_of_rules = rules.dict_of_rules
            dict_of_contributions = rules.dict_of_contributions_to_score_class

            dict_of_uncoding = {}
            for key, value in dict_of_coding.iteritems():
                dict_of_uncoding[str(value)] = key

            dict_of_stats = {}
            for key in dict_of_rules.iterkeys():
                dict_of_stats[key] = [dict_of_contributions[key]]
            for key, value in dict_of_rules.iteritems():
                dict_of_stats[key].append(self.trie.support_t(value, label))

            ans = []
            for key, value in dict_of_rules.iteritems():
                ans.append((self._rule_uncoding(value, dict_of_uncoding), dict_of_stats[key]))
            return ans
            # print dict_of_rules
            # print dict_of_contributions
        else:
            print("You need to train model!")

    def _classify_object(self, object_to_classification, silence=True):
        score_for_class = [0 for _ in xrange(self.number_of_classes)]

        rules_from_class = [[] for _ in xrange(self.number_of_classes)]

        for i in xrange(self.number_of_classes):
            for rule_id, rule in self.rules_class[i].dict_of_rules.items():
                rule_len = len(rule)
                if rule == object_to_classification[0: rule_len]:
                    rules_from_class[i].append(rule_id)

            id_to_remove_from_class = [[] for i in xrange(self.number_of_classes)]
            for rule_id1 in rules_from_class[i]:
                for rule_id2 in rules_from_class[i]:
                    if rule_id1 != rule_id2 and\
                            (rule_id1 not in id_to_remove_from_class[i]) and\
                            (rule_id2 not in id_to_remove_from_class[i]):
                        rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                        rule2 = self.rules_class[i].dict_of_rules[rule_id2]
                        if rule1[0: len(rule2)] == rule2:
                            id_to_remove_from_class[i].append(rule_id2)
                        elif rule2[0: len(rule1)] == rule1:
                            id_to_remove_from_class[i].append(rule_id1)
        for i in xrange(self.number_of_classes):
            for j in xrange(self.number_of_classes):
                for rule_id1 in rules_from_class[i]:
                    for rule_id2 in rules_from_class[j]:
                        if rule_id1 != rule_id2 and\
                                (rule_id1 not in id_to_remove_from_class[i]) and\
                                (rule_id2 not in id_to_remove_from_class[j]):
                            rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                            rule2 = self.rules_class[j].dict_of_rules[rule_id2]
                            if rule1[0: len(rule2)] == rule2:
                                id_to_remove_from_class[j].append(rule_id2)
                            elif rule2[0: len(rule1)] == rule1:
                                id_to_remove_from_class[i].append(rule_id1)

        for i in xrange(self.number_of_classes):
            # print(id_to_remove_from_class[i])
            for id_to_del in id_to_remove_from_class[i]:
                # print(rules_from_class[i], id_to_del)
                rules_from_class[i].remove(id_to_del)

        for i in xrange(self.number_of_classes):
            for rule in rules_from_class[i]:
                if rule in self.rules_class[i].dict_of_contributions_to_score_class:
                    score_for_class[i] += self.rules_class[i].dict_of_contributions_to_score_class[rule]
        if not silence:
            print(rules_from_class)
        if max(score_for_class) != 0:
            ans_class = np.argmax(score_for_class)

            return ans_class
        else:
            return -1

    def _classify_object_score(self, object_to_classification, silence=True):
        score_for_class = [0 for _ in xrange(self.number_of_classes)]

        rules_from_class = [[] for _ in xrange(self.number_of_classes)]

        for i in xrange(self.number_of_classes):
            for rule_id, rule in self.rules_class[i].dict_of_rules.items():
                rule_len = len(rule)
                if rule == object_to_classification[0: rule_len]:
                    rules_from_class[i].append(rule_id)

            id_to_remove_from_class = [[] for i in xrange(self.number_of_classes)]
            for rule_id1 in rules_from_class[i]:
                for rule_id2 in rules_from_class[i]:
                    if rule_id1 != rule_id2 and\
                            (rule_id1 not in id_to_remove_from_class[i]) and\
                            (rule_id2 not in id_to_remove_from_class[i]):
                        rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                        rule2 = self.rules_class[i].dict_of_rules[rule_id2]
                        # print(rule_id1)
                        # print(rule_id2)
                        # print(rule1)
                        # print(rule2)
                        if rule1[0: len(rule2)] == rule2:
                            id_to_remove_from_class[i].append(rule_id2)
                        elif rule2[0: len(rule1)] == rule1:
                            id_to_remove_from_class[i].append(rule_id1)

        for i in xrange(self.number_of_classes):
            for j in xrange(self.number_of_classes):
                for rule_id1 in rules_from_class[i]:
                    for rule_id2 in rules_from_class[j]:
                        if rule_id1 != rule_id2 and\
                                (rule_id1 not in id_to_remove_from_class[i]) and\
                                (rule_id2 not in id_to_remove_from_class[j]):
                            rule1 = self.rules_class[i].dict_of_rules[rule_id1]
                            rule2 = self.rules_class[j].dict_of_rules[rule_id2]
                            if rule1[0: len(rule2)] == rule2:
                                id_to_remove_from_class[j].append(rule_id2)
                            elif rule2[0: len(rule1)] == rule1:
                                id_to_remove_from_class[i].append(rule_id1)

        for i in xrange(self.number_of_classes):
            # print(id_to_remove_from_class[i])
            for id_to_del in id_to_remove_from_class[i]:
                # print(rules_from_class[i], id_to_del)
                rules_from_class[i].remove(id_to_del)

        for i in xrange(self.number_of_classes):
            for rule in rules_from_class[i]:
                if rule in self.rules_class[i].dict_of_contributions_to_score_class:
                    score_for_class[i] += self.rules_class[i].dict_of_contributions_to_score_class[rule]

        if not silence:
            print(rules_from_class)

        return score_for_class
