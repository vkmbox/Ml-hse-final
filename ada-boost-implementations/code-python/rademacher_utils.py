import numpy as np
import random
from ada_boost_standard_v1 import AdaBoostStandardClassifier_v1;

def calc_rademacher_sample(vectors_H, vector_rad):
    return max(np.inner(vectors_H, vector_rad))/len(vector_rad)

def calc_rademacher_biclassifiers(X, subset_size):
    samples_count, features_count = X.shape[0], X.shape[1]
    ind = np.argsort(X, axis=0)
    vectors_H = []
    for feature in range(features_count):
        for sample in range(samples_count):
            indices = ind[0:sample,feature]
            vector_H = np.array([-1]*samples_count)
            vector_H[indices] = 1
            vectors_H.append(vector_H)
            vectors_H.append(-1*vector_H)

    #vectors_H = [clf.classify(X) for clf in ensemble_classifiers]
    emperical_sum, subsets_number = 0.0, 2**samples_count
    if subsets_number > subset_size:
        for _ in range(subset_size):
            vector_rad = [1 if random.random() < 0.5 else -1 for _ in range(samples_count)]
            emperical_sum += calc_rademacher_sample(vectors_H, vector_rad)
    else:
        for num in range(subsets_number):
            vector_rad = -1 * np.ones(samples_count, np.int32)
            buff = "{0:b}".format(num)
            for pos in range(1, len(buff) + 1):
                vector_rad[-pos] = (1 if buff[-pos] == "1" else -1)
            #vector_rad = [int(smb) for smb in range(len(buff))]
            emperical_sum += calc_rademacher_sample(vectors_H, vector_rad)

    return emperical_sum/min(subset_size, subsets_number)

def calc_margin_loss(X, y, clf, rho):
    samples_count = X.shape[0]
    def F(arg, rho):
        if arg <= 0:
            return 1
        elif arg <= rho:
            return 1 - (arg/rho)
        else:
            return 0
    buff = np.multiply(y, clf(X))
    return sum([F(value, rho) for value in buff])/samples_count

if __name__ == '__main__':
    vectors_H = np.array([[1,1], [1,2]])
    assert calc_rademacher_sample(vectors_H, np.array([1,1])) == 3/2
    assert calc_rademacher_sample(vectors_H, np.array([1,-1])) == 0
    assert calc_rademacher_sample(vectors_H, np.array([-1,1])) == 1/2
    assert calc_rademacher_sample(vectors_H, np.array([-1,-1])) == -1

    X_train = np.array([[0.6476239, -0.81753611, -1.61389785, -0.21274028],
           [-2.3748206 ,  0.82768797, -0.38732682, -0.30230275],
           [ 1.51783379,  1.22140561, -0.51080514, -1.18063218],
           [-0.98740462,  0.99958558, -1.70627019,  1.9507754],
           [-1.43411205,  1.50037656, -1.04855297, -1.42001794],
           [ 0.29484027, -0.79249401, -1.25279536,  0.77749036]])
    
    y_train = np.array([1, -1, -1, 1, 1, -1])
    clf = AdaBoostStandardClassifier_v1(n_estimators=150)
    result, history = clf.fit(X_train, y_train, trace=True)
    print("Rademacher:", calc_rademacher_biclassifiers(X_train, 64)) #, clf.ensemble_classifiers
    print("Margin loss:"
        , calc_margin_loss(X_train, y_train, clf.get_esemble_result, clf.get_margin_l1(X_train)))

