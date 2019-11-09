import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class CountRegression:
    """Count regression.

    Arguments:
       lam (float): regaularization parameter lambda
    """
    def __init__(self, lam):
        self.reg_param = lam


    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Real-valued output vector for training.

        Notes: This function must set member variables such that a subsequent call
        to get_params or predict uses the learned parameters, overwriting
        any parameter values previously set by calling set_params.

        """
        """print(X)
        print(X.shape)
        print(X.tolist())
        print(y)
        print(y.shape)
        print(y.tolist())"""
        self.set_params(np.zeros((X.shape[1], 1)), 0)
        [w, b] = self.get_params()
        wb = np.append(w, b)

        result = fmin_l_bfgs_b(self.objective, x0=wb, args=(X, y), disp=10, fprime=self.objective_grad)
        self.set_params(result[0][:-1], result[0][-1])
        print('[w, b]', end=": ")
        print(self.get_params())

    def predict(self, X):
        """Predict using the odel.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,): predicted values
        """
        [w, b] = self.get_params()
        y = np.zeros((X.shape[0],1))

        for i in range(X.shape[0]):
            y[i] = np.dot(w, X[i][:]) + b

        return y


    def objective(self, wb, X, y):
        """Compute the objective function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            objective (float):
                the objective function evaluated on wb=[w,b] and the data X,y..
        """

        w = wb[:-1]
        b = wb[-1]

        obj = 0.0
        for i in range(X.shape[0]):
            obj = obj + np.dot(y[i], np.dot(w, X[i][:]) + b) + (y[i] + 1)*np.log(1 + np.exp(-(np.dot(w, X[i][:]) + b)))

        obj = obj + self.reg_param*np.linalg.norm(w)**2 + self.reg_param*b*b
        print(obj)

        return obj

    def objective_grad(self, wb, X, y):
        """Compute the derivative of the objective function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            objective_grad (ndarray, shape = (n_features + 1,)):
                derivative of the objective function with respect to wb=[w,b].
        """

        grad_wb = np.zeros(wb.shape)
        w = wb[:-1]
        b = wb[-1]

        grad_w = np.zeros(w.shape)
        grad_b = 0.0

        for n in range(X.shape[0]):
            exp = np.exp(-(np.dot(w, X[n][:]) + b))
            grad_w = grad_w + np.dot(y[n], X[n][:]) + np.divide(np.dot(np.dot((y[n] + 1), exp), -X[n][:]), 1 + exp)
        grad_w = grad_w + 2*self.reg_param*w

        for n in range(X.shape[0]):
            exp = np.exp(-(np.dot(w, X[n][:]) + b))
            grad_b = grad_b + y[n] + np.divide(np.dot(np.dot((y[n] + 1), exp), -1), 1 + exp)
        grad_b = grad_b + 2*self.reg_param*b

        grad_wb[:-1] = grad_w
        grad_wb[-1] = grad_b

        print(grad_wb)

        return grad_wb

    def get_params(self):
        """Get learned parameters for the model. Assumed to be stored in
           self.w, self.b.

        Returns:
            A tuple (w,b) where w is the learned coefficients (ndarray, shape = (n_features,))
            and b is the learned bias (float).
        """
        return [self.w, self.b]

    def set_params(self, w, b):
        """Set the parameters of the model. When called, this
           function sets the model parameters tha are used
           to make predictions. Assumes parameters are stored in
           self.w, self.b.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
        """
        self.w = w
        self.b = b

    def avg_pred_log(self, X, y):
        [w, b] = self.get_params()
        obj = 0.0
        for i in range(X.shape[0]):
            obj = obj + np.dot(y[i], np.dot(w, X[i][:]) + b) + (y[i] + 1)*np.log(1 + np.exp(-(np.dot(w, X[i][:]) + b)))

        avg_pred = obj/X.shape[0]

        return avg_pred

def main():

    data = np.load("data/count_data.npz")
    X_train=data['X_train']
    X_test=data['X_test']
    Y_train=data['Y_train']
    Y_test=data['Y_test']

    #Define and fit model
    cr  = CountRegression(1e-4)
    cr.fit(X_train,Y_train)
    print('avg pred log ', end=":")
    print(cr.avg_pred_log(X_test, Y_test))

if __name__ == '__main__':
    main()
