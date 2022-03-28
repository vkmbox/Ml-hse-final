import numpy as np
import scipy.optimize as opt

from datetime import datetime
from collections import defaultdict
from decision_stump_continuous import DecisionStumpContinuous as stump

class AdaBoostLinear_v1:
    
    def __init__(self, tolerance=1e-10):
        #self.n_estimators = -1
        self.tolerance = tolerance
        self.signs = (-1, 1)
        self.ensemble_coeffs = []
        self.ensemble_thresholds = []
        self.sample_size = None
        #self.ensemble_signs = []
        #self.ensemble_fnumbers = []

    def fit(self, X, y, allow_nonseparable=False, trace=False):
        self.ensemble_coeffs = []
        self.ensemble_thresholds = []
        #time_start = datetime.now()
        self.sample_size, features_count = X.shape[0], X.shape[1]
        n_estimators = 2*self.sample_size * features_count
        ind = np.argsort(X, axis=0)
        order = self.get_order_columnwise(ind)
        c = np.zeros(n_estimators + 1)
        c[-1] = -1
        a_ub = np.zeros((self.sample_size, n_estimators + 1))
        a_ub[:, -1] = 1
        feature_step, order_step = len(self.signs)*self.sample_size, len(self.signs)
        for sample_number in range(self.sample_size):
            for feature_number in range(features_count):
                for order_number in range(self.sample_size):
                    for sign_number in range(len(self.signs)):
                        sign = self.signs[sign_number]
                        h_k = sign if order[sample_number, feature_number] > order_number else -sign
                        a_ub[sample_number, feature_step*feature_number+order_step*order_number+sign_number] \
                            = -h_k * y[sample_number]

        b_ub = [0]*self.sample_size
        a_eq = np.ones((1, n_estimators+1))
        a_eq[0, -1] = 0
        b_eq = [1]
        xk_bounds = (0, None)
        xt_bounds = (None, None)
        bounds = [xk_bounds]*n_estimators
        bounds.append(xt_bounds)
        res = opt.linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds)

        coeffs, message, success = res.x, res.message, res.success

        if not success:
            return False, message
        if success and not allow_nonseparable and coeffs[-1] < 0:
            return False, "Failed to split data"

        self.ensemble_coeffs = coeffs[:-1]
        tmp = np.take_along_axis(X, ind, axis=0)
        tmp = np.repeat(tmp, repeats=2, axis=0)
        self.ensemble_thresholds = tmp.flatten('F')
        #self.ensemble_signs = [-1, +1]*n_estimators
        if coeffs[-1] < 0:
            message += " Training set is nonseparable."
        return True, message

    def predict(self, X):
        sample_size_step, features_count, n_estimators = self.sample_size * 2, X.shape[1], len(self.ensemble_coeffs)
        buffer = np.zeros(X.shape[0])
        for coeff, number, threshold, sign in \
                zip(self.ensemble_coeffs, range(n_estimators), self.ensemble_thresholds, self.signs*(int(n_estimators/2))):
            feature_number = number // sample_size_step
            buffer += coeff*np.array(stump.get_classification(X, feature_number, threshold, sign))
            
        return np.sign(buffer)

    '''
    def log(self, trace, history, minimal_error, d_t, time_start):
        if trace:
            history['time'].append((datetime.now() - time_start).total_seconds())
            history['error'].append(minimal_error)
            history['d_t'].append(d_t)
    '''
    def get_order_columnwise(self, ind):
        result = np.zeros_like(ind)
        for column in range(ind.shape[1]):
            for row in range(ind.shape[0]):
                result[ind[row, column], column] = row
        return result

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

    '''
    X_train = np.array([[0.1, 0.3],
                        [0.2, 0.1],
                        [0.3, 0.2]])


    X_train = np.array([[0.3],
                        [0.1],
                        [0.2]])

    X_train = np.array([[0.1],
                        [0.2],
                        [0.3]])

    y_train = np.array([1, 1, -1])
    '''

    clf = AdaBoostLinear_v1()
    result, message = clf.fit(X_train, y_train, trace=True)
    print(result, message)
    y_pred = clf.predict(X_train[0:2])
    print(y_pred)
    print(X_train[0:2])
#    print(clf.print_ensemble())

