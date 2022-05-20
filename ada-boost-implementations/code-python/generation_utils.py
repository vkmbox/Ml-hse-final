import numpy as np

def make_classification_normal(n_features, n_samples, rho=1e-1):
    x_data = np.random.multivariate_normal(np.zeros(n_features), np.eye(n_features), n_samples)
    return make_separable_labels(x_data, rho)

def make_separable_labels(x_data, rho=1e-1):
    n_samples, n_features = x_data.shape[0], x_data.shape[1]
    buff = np.random.uniform(-1, 1, n_features)
    v_normal = buff/np.linalg.norm(buff)
    y_data = np.sign(np.squeeze(np.matmul(x_data, v_normal))).astype(int)
    x_shift = np.tile(v_normal * rho, (n_samples,1))
    x_shift = np.multiply(x_shift, y_data[:, np.newaxis])
    return x_data + x_shift, y_data, v_normal

if __name__ == '__main__':
    x, y, _ = make_classification_normal(2, 5, rho=0.2)
    print(x, y)
