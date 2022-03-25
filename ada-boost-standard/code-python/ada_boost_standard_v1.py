import numpy as np
from collections import defaultdict
from datetime import datetime

class DecisionStumpContinuous:
    """
    DecisionStump-Continuous classifier.
    Classifies: if value > threshold then sign else -1 * sign, 
    where sign = {-1,+1}
    """
    
    def __init__(self, feature_number, sign, threshold):
        self.feature_number = feature_number
        self.sign = sign
        self.threshold = threshold
        
    def classify(self, X):
        feature = X[:, self.feature_number]
        return [self.sign if value > self.threshold else -self.sign for value in feature]
        
    def get_error(feature, y, d_t, sign, threshold):
        error = 0
        for arg, fn, weight in zip(feature, y, d_t):
            val = sign if arg > threshold else -1*sign
            error += abs(val-fn)*weight
            
        return error


class AdaBoostStandardClassifier:
    
    def __init__(self, n_estimators=50, tolerance=1e-10):
        self.n_estimators = n_estimators
        self.tolerance = tolerance
        self.signs = (-1, 1)
        self.ensemble = []

    def fit(self, X, y, sample_weigh = None, trace=False):
        time_start = datetime.now()
        sample_size = len(y)
        d_t = sample_weigh if sample_weigh is not None else np.full(sample_size, 1/sample_size)
        history = defaultdict(list) if trace else None
        ##d_t1 =  None
        for _ in range(self.n_estimators):
            minimal_error, decision_stump = self.get_decision_stump(X, y, d_t)
            if minimal_error >= 0.5:
                return 'error_level_exceeded', history
            alpha_t = 0.5 * np.log((1-minimal_error)/(minimal_error+self.tolerance))
            self.ensemble.append((alpha_t, decision_stump))
            self.log(trace, history, minimal_error, d_t, time_start)
            if minimal_error == 0:
                return 'error_free_classifier_found', history
            #d_t = np.dot(d_t, np.exp(-1*alpha_t*np.dot(y, decision_stump.classify(X))))
            d_t = np.multiply(d_t, np.exp(-alpha_t * np.multiply(y, decision_stump.classify(X))))
            d_t = d_t/(np.sum(d_t)+self.tolerance)
            
        return 'iterations_exceeded', history
    
    def predict(self, X):
        sample_size = X.shape[0]
        buffer = np.zeros(sample_size)
        for elm in self.ensemble:
            buffer += elm[0]*np.array(elm[1].classify(X))
            
        return np.sign(buffer)
    
    def get_decision_stump(self, X, y, d_t):
        features_count = X.shape[1]
        minimal_error, minimal_feature, minimal_sign, minimal_threshold \
            = None, None, None, None
        for feature_number in range(features_count):
            feature = X[:, feature_number]
            for threshold in feature:
                for sign in self.signs:
                    current_error = DecisionStumpContinuous.get_error\
                        (feature, y, d_t, sign, threshold)
                    if minimal_error is None or minimal_error > current_error:
                        minimal_error, minimal_feature, minimal_sign, minimal_threshold\
                            = current_error, feature_number, sign, threshold

        return minimal_error, DecisionStumpContinuous(minimal_feature, minimal_sign, minimal_threshold)

    def log(self, trace, history, minimal_error, d_t, time_start):
        if trace:
            history['time'].append((datetime.now() - time_start).total_seconds())
            history['error'].append(minimal_error)
            history['d_t'].append(d_t)

if __name__ == '__main__':
    X_train = np.array([[0.6476239, -0.81753611, -1.61389785, -0.21274028],
           [-2.3748206 ,  0.82768797, -0.38732682, -0.30230275],
           [ 1.51783379,  1.22140561, -0.51080514, -1.18063218],
           [-0.98740462,  0.99958558, -1.70627019,  1.9507754 ],
           [-1.43411205,  1.50037656, -1.04855297, -1.42001794],
           [ 0.29484027, -0.79249401, -1.25279536,  0.77749036]])
    
    y_train = np.array([1, -1, -1, 1, 1, -1])
    
    clf = AdaBoostStandardClassifier(n_estimators=100)
    result, history = clf.fit(X_train, y_train, trace=True)
    print(result)
    print(history['error'])
    print(history['d_t'])
    y_pred = clf.predict(X_train)
    print(y_pred)
    #print(clf.ensemble)
