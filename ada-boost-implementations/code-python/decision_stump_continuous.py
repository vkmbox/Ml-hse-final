class DecisionStumpContinuous:
    """
    DecisionStump-Continuous classifier.
    Classifies: if value > threshold then sign else -1 * sign, 
    where sign = {-1,+1}
    """
    
    def __init__(self, feature_number, sign, threshold):
        self.feature_number = feature_number
        self.threshold = threshold
        self.sign = sign

    def classify(self, X):
        #feature = X[:, self.feature_number]
        #return [self.sign if value > self.threshold else -self.sign for value in X[:, self.feature_number]]
        return self.get_classification(X, self.feature_number, self.threshold, self.sign)

    @staticmethod
    def get_classification(X, feature_number, threshold, sign):
        return [DecisionStumpContinuous.get_classification_scalar(value, threshold, sign) for value in X[:, feature_number]]
        #return [sign if value > threshold else -sign for value in X[:, feature_number]]

    @staticmethod
    def get_classification_scalar(x_scalar, threshold, sign):
        return sign if x_scalar > threshold else -sign

    @staticmethod
    def get_error(feature, y, d_t, sign, threshold):
        error = 0
        for arg, fn, weight in zip(feature, y, d_t):
            val = sign if arg > threshold else -1*sign
            error += abs((val-fn)/2)*weight
            
        return error

    def str(self):
        return "feature_number: {}, sign: {}, threshold: {}".format(self.feature_number, self.sign, self.threshold)
