import numpy as np
from py4j.java_gateway import JavaGateway


class AdaBoostJavaClassifier_v1:
    def __init__(self, path):
        self.path = path
        self.gateway = JavaGateway()
        self.classifier = self.gateway.entry_point

    def getX(self, X):
        samples_count, features_count = X.shape[0], X.shape[1]
        dataX = self.gateway.new_array(self.gateway.jvm.double, samples_count, features_count)
        for row in range(samples_count):
            for col in range(features_count):
                dataX[row][col] = X[row, col].item()
        return dataX

    def getY(self, y):
        samples_count = y.shape[0]
        dataY = self.gateway.new_array(self.gateway.jvm.int, samples_count)
        for row in range(samples_count):
            dataY[row] = y[row].item()
        return dataY
    '''
    def fit(self, X, y):
        dataX, dataY = self.getX(X), self.getY(y)
        return self.classifier.fit(dataX, dataY)
    '''
    def fit(self, X, y, n_estimators=150):
        dataX, dataY = self.getX(X), self.getY(y)
        return self.classifier.fit(dataX, dataY, n_estimators)

    def predict(self, X):
        dataX = self.getX(X)
        return np.array(self.classifier.predict(dataX))

    def get_margin_l1(self, X):
        dataX = self.getX(X)
        return self.classifier.getMarginL1(dataX)

    def calc_rademacher_biclassifiers(self, X, subset_size):
        dataX = self.getX(X)
        return self.classifier.calculateRademacherForBiclassifiers(dataX, subset_size)

    def calc_margin_loss(self, X, y, rho):
        dataX, dataY = self.getX(X), self.getY(y)
        return self.classifier.calculateMarginLoss(dataX, dataY, rho)

if __name__ == '__main__':
    classpath = 'd:/Projects-my/ml/Ml-hse-final/ada-boost-implementations/code-python/adaboost_java/lib-adaboost.jar'
    X_train = np.array([[0.6476239, -0.81753611, -1.61389785, -0.21274028],
                       [-2.37482060,  0.82768797, -0.38732682, -0.30230275],
                       [ 1.51783379,  1.22140561, -0.51080514, -1.18063218],
                       [-0.98740462,  0.99958558, -1.70627019,  1.9507754 ],
                       [-1.43411205,  1.50037656, -1.04855297, -1.42001794],
                       [ 0.29484027, -0.79249401, -1.25279536,  0.77749036]])

    y_train = np.array([1, -1, -1, 1, 1, -1])
    clf = AdaBoostJavaClassifier_v1(path=classpath)
    result = clf.fit(X_train, y_train, 146)
    print(result)
    y_pred = clf.predict(X_train)
    assert (y_train==y_pred).all(), 'Wrong answer'
    print("Margin L1:", clf.get_margin_l1(X_train))
