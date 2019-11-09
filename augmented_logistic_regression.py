import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn import linear_model

class AugmentedLogisticRegression:
    """Logistic regression with optimized centering.

    Arguments:
        lambda(float): regularization parameter lambda (default: 0)
    """

    def __init__(self, lmbda=0):
        self.reg_param = lmbda  # regularization parameter (lambda)

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training output vector. Each entry is either -1 or 1.

        Notes: This function must set member variables such that a subsequent call
        to get_params or predict uses the learned parameters, overwriting
        any parameter values previously set by calling set_params.
        """
        self.set_params(np.zeros((X.shape[1], 1)), np.zeros((X.shape[1], 1)), 1)
        [w, c, b] = self.get_params()
        wcb = np.append(np.append(w,c),b)
        result = fmin_l_bfgs_b(self.objective, x0=wcb, args=(X, y), disp=10, fprime=self.objective_grad)
        self.set_params(result[0][: X.shape[1]], result[0][X.shape[1]: -1], result[0][-1])
        print(self.get_params())

        return self

    def predict(self, X):
        """Predict class labels for samples in X based on current parameters.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values in {-1, +1}.
        """
        [w, c, b] = self.get_params()
        y = np.zeros((X.shape[0],1))

        for i in range(X.shape[0]):
            p_pred = np.divide(1, 1 + np.exp(np.multiply(-1, np.dot(X[i][:] - c, w) + b)))
            n_pred = np.divide(1, 1 + np.exp(np.multiply(1, np.dot(X[i][:] - c, w) + b)))
            y[i] = 1 if p_pred > n_pred else -1

        y = y.reshape((X.shape[0],))
        return y

    def objective(self, wcb, X, y):
        """Compute the learning objective function

        Arguments:
            wcb (ndarray, shape = (2*n_features + 1,)):
                concatenation of the coefficient, centering, and bias parameters
                wcb = [w, c, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            objective (float):
                the objective function evaluated at [w, b, c] and the data X, y.
        """
        w = wcb[:X.shape[1]]
        c = wcb[X.shape[1]:-1]
        b = wcb[-1]

        obj = 0.0
        for i in range(X.shape[0]):
            obj = obj + np.log(1 + np.exp(np.dot(-y[i], np.dot(w, (X[i][:] - c))+b)))

        obj = obj + self.reg_param*np.linalg.norm(w)**2 + self.reg_param*np.linalg.norm(c)**2 + self.reg_param*b*b
        print(type(obj))

        return obj

    def objective_grad(self, wcb, X, y):
        """Compute the gradient of the learning objective function

        Arguments:
            wcb (ndarray, shape = (2*n_features + 1,)):
                concatenation of the coefficient, centering, and bias parameters
                wcb = [w, c, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            objective_grad (ndarray, shape = (2*n_features + 1,)):
                gradient of the objective function with respect to [w,c,b].
        """
        #assert wcb.shape == (wcb.shape[0],)
        grad_wcb = np.zeros(wcb.shape)
        w = wcb[:X.shape[1]]
        c = wcb[X.shape[1]:-1]
        b = wcb[-1]

        grad_w = np.zeros(w.shape)
        grad_c = np.zeros(c.shape)
        grad_b = 0.0

        for n in range(X.shape[0]):
            exp = np.exp(-y[n]*(np.dot((X[n][:] - c), w)+b))
            grad_w = grad_w + np.divide(exp*(-y[n])*(X[n][:] - c), 1 + exp)
        grad_w = grad_w + 2*self.reg_param*w

        for n in range(X.shape[0]):
            exp = np.exp(-y[n]*(np.dot((X[n][:] - c), w)+b))
            grad_c = grad_c + np.divide(exp*(y[n])*w, 1 + exp)
        grad_c = grad_c + 2*self.reg_param*c

        for n in range(X.shape[0]):
            exp = np.exp(-y[n]*(np.dot((X[n][:] - c), w)+b))
            grad_b = grad_b + np.divide(exp*(-y[n]), 1 + exp)
        grad_b = grad_b + 2*self.reg_param*b

        grad_wcb[:X.shape[1]] = grad_w
        grad_wcb[X.shape[1]:-1] = grad_c
        grad_wcb[-1] = grad_b

        print(grad_wcb.shape)

        return grad_wcb

    def get_params(self):
        """Get parameters for the model.

        Returns:
            A tuple (w,c,b) where w is the learned coefficients (ndarray, shape = (n_features,)),
            c  is the learned centering parameters (ndarray, shape = (n_features,)),
            and b is the learned bias (float).
        """
        return [self.w, self.c, self.b]

    def set_params(self, w, c, b):
        """Set the parameters of the model.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficients
            c (ndarray, shape = (n_features,)): centering parameters
            b (float): bias
        """
        self.w = w
        self.c = c
        self.b = b

def main():
    np.random.seed(0)

    train_X = np.load('data/q2_train_X.npy')
    train_y = np.load('data/q2_train_y.npy')
    test_X = np.load('data/q2_test_X.npy')
    test_y = np.load('data/q2_test_y.npy')

    lr = AugmentedLogisticRegression(lmbda = 1e-6)
    lr.fit(train_X, train_y)
    ##lr.avg_error(train_X, train_y)
    ##lr.compare_model(train_X, train_y)
    #lr.set_params(np.zeros((test_X.shape[1], 1)), np.zeros((test_X.shape[1], 1)), 1)
    #lr.predict(test_X)
    #print(test_y.tolist())

if __name__ == '__main__':
    main()
