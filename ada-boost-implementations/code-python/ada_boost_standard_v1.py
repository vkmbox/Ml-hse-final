import numpy as np

from datetime import datetime
from collections import defaultdict
from decision_stump_continuous import DecisionStumpContinuous

class AdaBoostStandardClassifier_v1:
    
    def __init__(self, n_estimators=50, tolerance=1e-10):
        self.n_estimators = n_estimators
        self.tolerance = tolerance
        self.signs = (-1, 1)
        self.ensemble_alphas = []
        self.ensemble_classifiers = []
        #self.ensemble = []

    def fit(self, X, y, sample_weigh = None, trace=False):
        self.ensemble_alphas, self.ensemble_classifiers = [], []
        time_start = datetime.now()
        sample_size = len(y)
        d_t = sample_weigh if sample_weigh is not None else np.full(sample_size, 1/sample_size)
        history = defaultdict(list) if trace else None
        for _ in range(self.n_estimators):
            minimal_error, decision_stump = self.get_decision_stump(X, y, d_t)
            if minimal_error >= 0.5:
                return 'error_level_exceeded', history
            alpha_t = 0.5 * np.log((1-minimal_error)/(minimal_error+self.tolerance))
            self.ensemble_alphas.append(alpha_t)
            self.ensemble_classifiers.append(decision_stump)
            self.log(trace, history, minimal_error, d_t, time_start)
            if minimal_error == 0:
                return 'error_free_classifier_found', history
            d_t = np.multiply(d_t, np.exp(-alpha_t * np.multiply(y, decision_stump.classify(X))))
            d_t_sum = np.sum(d_t)+self.tolerance
            d_t = d_t/d_t_sum
            
        return 'iterations_exceeded', history

    def get_decision_stump(self, X, y, d_t):
        samples_count, features_count = X.shape[0], X.shape[1]
        minimal_error, minimal_feature, minimal_sign, minimal_threshold \
            = None, None, None, None
        for feature_number in range(features_count):
            #feature = X[:, feature_number]
            #for threshold in feature:
            for sample_number in range(samples_count):
                threshold = X[sample_number, feature_number]
                for sign in self.signs:
                    current_error = DecisionStumpContinuous.get_error\
                        (X, y, feature_number, d_t, sign, threshold)
                        #(feature, y, d_t, sign, threshold)
                    if minimal_error is None or minimal_error > current_error:
                        minimal_error, minimal_feature, minimal_sign, minimal_threshold\
                            = current_error, feature_number, sign, threshold

        return minimal_error, DecisionStumpContinuous(minimal_feature, minimal_sign, minimal_threshold)

    def log(self, trace, history, minimal_error, d_t, time_start):
        if trace:
            history['time'].append((datetime.now() - time_start).total_seconds())
            history['error'].append(minimal_error)
            history['d_t'].append(d_t)

    def get_esemble_result(self, X):
        sample_size = X.shape[0]
        buffer = np.zeros(sample_size)
        for alpha, classifier in zip(self.ensemble_alphas, self.ensemble_classifiers): #self.ensemble:
            buffer += alpha*np.array(classifier.classify(X))

        return buffer #!!!Not divided by L1 norm of alphas!

    def predict_raw(self, X):
        buffer = self.get_esemble_result(X)
        alpha_modulo = sum([abs(alpha) for alpha in self.ensemble_alphas]) + self.tolerance
        return buffer/alpha_modulo

    def predict(self, X):
        return np.sign(self.get_esemble_result(X))

    def get_margin_l1(self, X):
        buffer = self.get_esemble_result(X)
        alpha_modulo = sum([abs(alpha) for alpha in self.ensemble_alphas]) + self.tolerance
        return min(abs(buffer))/alpha_modulo

    def print_ensemble(self):
        value = ""
        for alpha, classifier in zip(self.ensemble_alphas, self.ensemble_classifiers): #self.ensemble:
            value += "alpha={}, classifier=<{}>;".format(alpha, classifier.str())
        return value

if __name__ == '__main__':
    X_train = np.array([[0.6476239, -0.81753611, -1.61389785, -0.21274028],
           [-2.3748206 ,  0.82768797, -0.38732682, -0.30230275],
           [ 1.51783379,  1.22140561, -0.51080514, -1.18063218],
           [-0.98740462,  0.99958558, -1.70627019,  1.9507754],
           [-1.43411205,  1.50037656, -1.04855297, -1.42001794],
           [ 0.29484027, -0.79249401, -1.25279536,  0.77749036]])
    
    y_train = np.array([1, -1, -1, 1, 1, -1])

    clf = AdaBoostStandardClassifier_v1(n_estimators=100)
    result, history = clf.fit(X_train, y_train, trace=True)
    print(result)
    y_pred = clf.predict(X_train)
    assert (y_train==y_pred).all(), 'Wrong answer'
    print("Margin L1:", clf.get_margin_l1(X_train))
