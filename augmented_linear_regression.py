import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn import linear_model
#import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches

class AugmentedLinearRegression:
    """Augmented linear regression.

    Arguments:
        delta (float): the trade-off parameter of the loss
    """
    def __init__(self, delta):
        self.delta = delta

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


        """n_samples, n_features = X.shape
        w = np.random.rand(n_samples, n_features)
        print(w)"""
        self.set_params(np.zeros((X.shape[1], 1)), 0)
        [w, b] = self.get_params()
        wb = np.append(w, b)

        result = fmin_l_bfgs_b(self.objective, x0=wb, args=(X, y), disp=10, fprime=self.objective_grad)
        print(result[0])
        self.set_params(result[0][:-1], result[0][-1])

        return self

    def predict(self, X):
        """Predict using the linear model.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,): predicted values
        """
        [w, b] = self.get_params()
        y = np.matmul(X, w) + b

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
        self.set_params(wb[:-1], wb[-1])
        y_pred = list(self.predict(X))
        assert len(y) == len(y_pred)

        obj = 0.0
        for i in range(X.shape[0]):
            obj = obj + np.dot(self.delta**2, np.sqrt(1 + np.square(y[i] - y_pred[i])/self.delta**2) - 1)

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

        X0 = np.ones((X.shape[0], 1))
        X_new = np.hstack((X, X0))

        grad_wb = np.zeros((X_new.shape[1], 1))
        grad_wb = self.delta * np.matmul(X_new.T, np.divide(np.matmul(X_new, wb) - y, np.sqrt(self.delta**2 + (y - np.matmul(X_new, wb))**2)))
        print(grad_wb)
        print(wb)

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
           function sets the model parameters that are used
           to make predictions. Assumes parameters are stored in
           self.w, self.b.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
        """
        self.w = w
        self.b = b

    def plot_loss(self, X, y):

        fig1 = plt.gcf()

        delta1 = 0.1
        delta2 = 1
        delta3 = 10
        robust_loss1 = np.ndarray((X.shape[0],))
        robust_loss2 = np.ndarray((X.shape[0],))
        robust_loss3 = np.ndarray((X.shape[0],))
        sqr_loss = np.ndarray((X.shape[0],))

        errors = np.random.randn(X.shape[0], )
        #print(robust_loss1.shape)
        #print(errors.shape)
        for i in range(X.shape[0]):
            robust_loss1[i] = np.dot(delta1**2, np.sqrt(1 + np.square(errors[i])/delta1**2) - 1)
            robust_loss2[i] = np.dot(delta2**2, np.sqrt(1 + np.square(errors[i])/delta2**2) - 1)
            robust_loss3[i] = np.dot(delta3**2, np.sqrt(1 + np.square(errors[i])/delta3**2) - 1)
            sqr_loss[i] = np.power(errors[i], 2)

        plt.scatter(errors, robust_loss1, color=["#C23B23"], label='$\delta = 0.1$')
        plt.scatter(errors, robust_loss2, color=["#F39A27"], label='$\delta = 1$')
        plt.scatter(errors, robust_loss3, color=["#03C03C"], label='$\delta = 10$')
        plt.scatter(errors, sqr_loss, color=["#579ABE"], label='Squared Loss')
        #plt.legend([mpatches.Patch(color='#C23B23'), mpatches.Patch(color='#F39A27'), mpatches.Patch(color='#03C03C'), mpatches.Patch(color='#579ABE')], ('$\delta$ = 0.1', '$\delta$ = 1', '$\delta$ = 10', 'Standard Squared Loss'), prop={'size': 18})
        plt.legend(prop={'size': 22})
        plt.xlabel('Prediction Error $\epsilon$ = $y$ - $y\'$', fontsize=22)
        plt.ylabel('Value of the loss $L_\delta$($y$, $y\'$) and $L_{sqr}$($y$, $y\'$)', fontsize=22)
        plt.title('Robust Loss Function [$\delta$ $\in$ {0.1, 1, 10}] and Standard Squared Loss Function versus Prediction Error $\epsilon$ = $y$ - $y\'$', fontsize=22)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
        fig1.savefig('LossPlot.png')

    def compare_linreg(self, lin_model, X, y):
        print('slope w:', lin_model.coef_)
        print('intercept b:', lin_model.intercept_)
        MSE_robust = np.square(np.subtract(y, self.predict(X))).mean()
        MSE_linear = np.square(np.subtract(y, lin_model.predict(X))).mean()
        print("MSE with robust loss function with delta = 1: ", end="")
        print(MSE_robust)
        print("MSE with standard least squares linear function: ", end="")
        print(MSE_linear)

    def compare_predication(self, lin_model, X, y):
        y_pred_1 = self.predict(X)
        fig1 = plt.gcf();
        plt.scatter(X, y, color=["#8DCB46", "#8AF70C", "#C2FF00", "#DAF7A6"], label='Training Data')
        robust_line = plt.plot(X, y_pred_1, label='Regression Line for Robust Model')
        std_line = plt.plot(X, lin_model.predict(X), label='Regression Line for Standard Squared Model')
        plt.legend(prop={'size':22})
        plt.xlabel('Training Dataset X', fontsize=22)
        plt.ylabel('Training Dataset Y', fontsize=22)
        plt.title('X versus Y Training Dataset', fontsize=22)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
        fig1.savefig('Data.png')

def main():

    np.random.seed(0)
    train_X = np.load('data/q3_train_X.npy')
    train_y = np.load('data/q3_train_y.npy')

    lr = AugmentedLinearRegression(delta=1)
    lr.fit(train_X, train_y)

    lin_model = linear_model.LinearRegression()
    lin_model.fit(train_X, train_y)

    #lr.plot_loss(train_X, train_y)
    #lr.compare_linreg(lin_model, train_X, train_y)
    #lr.compare_predication(lin_model, train_X, train_y)

if __name__ == '__main__':
    main()
