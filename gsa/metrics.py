from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np


def accuracy_score_with_unclassified_objects(y, y_predict):
    y_new = []
    y_predict_new = []
    number_of_unclassified = 0

    for i in xrange(len(y)):
        if y_predict[i] != -1:
            y_new.append(y[i])
            y_predict_new.append(y_predict[i])
        else:
            number_of_unclassified += 1

    accuracy = accuracy_score(y_new, y_predict_new)

    return accuracy, number_of_unclassified


def confusion_matrix_with_unclassified(y, y_predict):
    y_new = []
    y_predict_new = []
    number_of_unclassified = 0

    for i in xrange(len(y)):
        if y_predict[i] != -1:
            y_new.append(y[i])
            y_predict_new.append(y_predict[i])
        else:
            number_of_unclassified += 1

    confusion_m = confusion_matrix(y_new, y_predict_new)

    return confusion_m, number_of_unclassified


def tpr_fpr_nonclass(y, y_predict):
    y_new = []
    y_predict_new = []
    number_of_unclassified = 0

    for i in xrange(len(y)):
        if y_predict[i] != -1:
            y_new.append(y[i])
            y_predict_new.append(y_predict[i])
        else:
            number_of_unclassified += 1

    confusion_m = confusion_matrix(y_new, y_predict_new)
    # print(confusion_m)
    if confusion_m != []:
        tp = confusion_m[1, 1]
        fp = confusion_m[0, 1]

        tn = confusion_m[0, 0]
        fn = confusion_m[1, 0]

        pos = tp + fn
        neg = tn + fp

        tpr = tp/float(pos)
        fpr = fp/float(neg)
    else:
        tpr = 0
        fpr = 0

    return tpr, fpr, number_of_unclassified


def tp_tn_fp_fn_ncp_ncn(y, y_predict):
    y_new = []
    y_predict_new = []
    number_of_p_unclassified = 0
    number_of_n_unclassified = 0

    for i in xrange(len(y)):
        if y_predict[i] != -1:
            y_new.append(y[i])
            y_predict_new.append(y_predict[i])
        else:
            if y[i] == 0:
                number_of_n_unclassified += 1
            elif y[i] == 1:
                number_of_p_unclassified += 1

    confusion_m = confusion_matrix(y_new, y_predict_new)
    # print(confusion_m)
    if confusion_m != []:
        tp = confusion_m[1, 1]
        fp = confusion_m[0, 1]

        tn = confusion_m[0, 0]
        fn = confusion_m[1, 0]

        pos = tp + fn
        neg = tn + fp

        tpr = tp/float(pos)
        fpr = fp/float(neg)
    else:
        tpr = 0
        fpr = 0

    # number_of_pos = float(sum(y))
    # number_of_neg = float(len(y) - sum(y))
    # ncpr = number_of_p_unclassified / number_of_pos
    # ncnr = number_of_n_unclassified / number_of_neg

    return tp, tn, fp, fn, number_of_p_unclassified, number_of_n_unclassified


def tpr_fpr_ncpr_ncnr(y, y_predict):
    y_new = []
    y_predict_new = []
    number_of_p_unclassified = 0
    number_of_n_unclassified = 0

    for i in xrange(len(y)):
        if y_predict[i] != -1:
            y_new.append(y[i])
            y_predict_new.append(y_predict[i])
        else:
            if y[i] == 0:
                number_of_n_unclassified += 1
            elif y[i] == 1:
                number_of_p_unclassified += 1

    confusion_m = confusion_matrix(y_new, y_predict_new)
    # print(confusion_m)
    if confusion_m != []:
        tp = confusion_m[1, 1]
        fp = confusion_m[0, 1]

        tn = confusion_m[0, 0]
        fn = confusion_m[1, 0]

        pos = tp + fn
        neg = tn + fp

        tpr = tp/float(pos)
        fpr = fp/float(neg)
    else:
        tpr = 0
        fpr = 0

    number_of_pos = float(sum(y))
    number_of_neg = float(len(y) - sum(y))
    ncpr = number_of_p_unclassified / number_of_pos
    ncnr = number_of_n_unclassified / number_of_neg

    return tpr, fpr, ncpr, ncnr


def f1_score_nonclass(y, y_predict):
    y_new = []
    y_predict_new = []
    number_of_unclassified = 0

    for i in xrange(len(y)):
        if y_predict[i] != -1:
            y_new.append(y[i])
            y_predict_new.append(y_predict[i])
        else:
            number_of_unclassified += 1

    f1 = f1_score(y_new, y_predict_new)

    return f1, number_of_unclassified


class CostValueAbstainingClassifiers(object):
    def __init__(self, cost_matrix=(1, 1/2.0)):
        self.mu = cost_matrix[0]
        self.nu = cost_matrix[1]

    def expected_cost(self, y_valid, y_predict):
        """
        C = ([C(P,p), C(P,n), C(P,a)]; [C(N,p), C(N, n), C(N,a)])
        In our case
        C =([0, 1, C'(P,a)]; [C'(N,p), 0, C'(P,a)])

        mu is a C'(N,p) - cost of fasle-positive
        nu is a C'(P,a) - cost of abstaining classified object
        :param y_valid: list of real labels
        :param y_predict: list of predicted labels
        :return: float cost value
        """
        P_n_P = 0
        P_p_N = 0
        P_a = 0
        P_P = sum(y_valid)
        P_N = len(y_valid) - sum(y_valid)
        n = float(len(y_valid))

        for i in xrange(len(y_valid)):
            if y_valid[i] == 1 and y_predict[i] == 0:
                P_n_P += 1
            elif y_valid[i] == 0 and y_predict[i] == 1:
                P_p_N += 1
            elif y_predict[i] == -1:
                P_a += 1
        P_n_P /= n
        P_p_N /= n
        P_a /= n
        P_P /= n
        P_N /= n
        cost = P_n_P + self.mu * P_p_N + self.nu * P_a

        return cost

if __name__ == '__main__':
    cost_counter = CostValueAbstainingClassifiers([1, 1/10.0])
    ans = [1, 1, 1, 0, 0, 0]
    pred = [-1, -1, -1, -1, -1, -1]
    pred1 = [-1, -1, 1, -1, -1, 1]

    print cost_counter.expected_cost(ans, pred)
    print cost_counter.expected_cost(ans, pred1)
