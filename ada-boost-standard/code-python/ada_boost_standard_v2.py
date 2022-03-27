import numpy as np

from datetime import datetime
from collections import defaultdict
from decision_stump_continuous import DecisionStumpContinuous


class AdaBoostStandardClassifier_v2:
    
    def __init__(self, n_estimators=50, tolerance=1e-10):
        self.n_estimators = n_estimators
        self.tolerance = tolerance
        self.signs = (-1, 1)
        self.ensemble = []

    def fit(self, X, y, sample_weigh = None, trace=False):
        #y = np.asarray(y)
        time_start = datetime.now()
        sample_size, features_count = len(y), X.shape[1]
        d_t = sample_weigh if sample_weigh is not None else np.full(sample_size, 1/sample_size)
        yy = np.tile(np.array([y]).transpose(), (1, features_count))
        history = defaultdict(list) if trace else None

        for _ in range(self.n_estimators):
            dd_t = np.tile(np.array([d_t]).transpose(), (1, features_count))
            minimal_error, decision_stump = self.get_decision_stump(X, yy, dd_t)
            if minimal_error >= 0.5:
                return 'error_level_exceeded', history
            alpha_t = 0.5 * np.log((1-minimal_error)/(minimal_error+self.tolerance))
            self.ensemble.append((alpha_t, decision_stump))
            self.log(trace, history, minimal_error, d_t, time_start)
            if minimal_error == 0:
                return 'error_free_classifier_found', history
            d_t = np.multiply(d_t, np.exp(-alpha_t * np.multiply(y, decision_stump.classify(X))))
            d_t = d_t/(np.sum(d_t)+self.tolerance)
            
        return 'iterations_exceeded', history
    
    def predict(self, X):
        sample_size = X.shape[0]
        buffer = np.zeros(sample_size)
        for elm in self.ensemble:
            buffer += elm[0]*np.array(elm[1].classify(X))
            
        return np.sign(buffer)
    
    def get_decision_stump(self, X, yym, dd_tm):
        measurements, features_count = X.shape[0], X.shape[1]
        minimal_error, minimal_feature, minimal_sign, minimal_threshold \
            = None, None, None, None

        ind = np.argsort(X, axis=0)
        xx = np.take_along_axis(X, ind, axis=0)
        yy = np.take_along_axis(yym, ind, axis=0)
        dd_t = np.take_along_axis(dd_tm, ind, axis=0)

        for feature_number in range(features_count):
            '''
            feature = XX[:, feature_number]
            permutation = ind[:, feature_number]
            yy = YY[:, feature_number] #np.take_along_axis(y, permutation, axis=0)
            dd_t = DD_T[:, feature_number] #np.take_along_axis(d_t, permutation, axis=0)
            '''
            for threshold in range(measurements):
                error_plus, error_minus = 0.0, 0.0
                for pos in range(measurements):
                    if pos <= threshold:
                        error_plus += dd_t[pos, feature_number] if yy[pos, feature_number] == 1 else 0
                        error_minus += dd_t[pos, feature_number] if yy[pos, feature_number] == -1 else 0
                    else:
                        error_plus += dd_t[pos, feature_number] if yy[pos, feature_number] == -1 else 0
                        error_minus += dd_t[pos, feature_number] if yy[pos, feature_number] == 1 else 0

                if error_plus <= error_minus and (minimal_error is None or error_plus < minimal_error):
                    minimal_error, minimal_feature, minimal_sign, minimal_threshold \
                        = error_plus, feature_number, +1, xx[threshold, feature_number]
                if error_minus < error_plus and (minimal_error is None or error_minus < minimal_error):
                    minimal_error, minimal_feature, minimal_sign, minimal_threshold \
                        = error_minus, feature_number, -1, xx[threshold, feature_number]

            '''
            for threshold in feature:
                for sign in self.signs:
                    current_error = DecisionStumpContinuous.get_error\
                        (feature, y, d_t, sign, threshold)
                    if minimal_error is None or minimal_error > current_error:
                        minimal_error, minimal_feature, minimal_sign, minimal_threshold\
                            = current_error, feature_number, sign, threshold
            '''

        return minimal_error, DecisionStumpContinuous(minimal_feature, minimal_sign, minimal_threshold)

    def log(self, trace, history, minimal_error, d_t, time_start):
        if trace:
            history['time'].append((datetime.now() - time_start).total_seconds())
            history['error'].append(minimal_error)
            history['d_t'].append(d_t)

    def print_ensemble(self):
        value = ""
        for elm in self.ensemble:
            value += "alpha={}, classifier=<{}>;".format(elm[0], elm[1].str())
        return value

if __name__ == '__main__':
    X_train = np.array([[0.6476239, -0.81753611, -1.61389785, -0.21274028],
                       [-2.37482060,  0.82768797, -0.38732682, -0.30230275],
                       [ 1.51783379,  1.22140561, -0.51080514, -1.18063218],
                       [-0.98740462,  0.99958558, -1.70627019,  1.9507754 ],
                       [-1.43411205,  1.50037656, -1.04855297, -1.42001794],
                       [ 0.29484027, -0.79249401, -1.25279536,  0.77749036]])
    
    y_train = np.array([1, -1, -1, 1, 1, -1])
    
    clf = AdaBoostStandardClassifier_v2(n_estimators=100)
    result, history = clf.fit(X_train, y_train, trace=True)
    print(result)
    print(history['error'])
    print(history['d_t'])
    y_pred = clf.predict(X_train)
    print(y_pred)
    print(clf.print_ensemble())
