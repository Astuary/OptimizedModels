import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt


class SVM:
    """SVC with subgradient descent training.

    Arguments:
        C: regularization parameter (default: 1)
        iterations: number of training iterations (default: 500)
    """
    def __init__(self, C=1, iterations=500):
        self.C = C
        self.iter = iterations

    def fit(self, X, y):
        """Fit the model using the training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Notes: This function must set member variables such that a subsequent call
        to get_params or predict uses the learned parameters, overwriting
        any parameter values previously set by calling set_params.

        """
        self.set_params(np.zeros((X.shape[1], 1)), 0)
        [w, b] = self.get_params()
        wb = np.append(w, b)

        f_min = np.inf
        wb_f = np.zeros(wb.shape)

        for i in range(self.iter):
            grad_wb = self.subgradient(wb, X, y)
            alpha = 0.002/np.sqrt(i + 1)
            wb_new = wb - alpha * grad_wb

            obj = self.objective(wb_new, X, y)
            if obj < f_min:
                f_min = obj
                wb_f = wb_new

            wb = wb_new

        print(wb_f.tolist())
        self.set_params(wb_f[:-1], wb_f[-1])

        print(self.objective(wb_f, X, y))
        print(self.classification_error(X, y))


    def objective(self, wb, X, y):
        """Compute the objective function for the SVM.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            obj (float): value of the objective function evaluated on X and y.
        """

        w = wb[:-1]
        b = wb[-1]

        obj = 0.0
        for n in range(X.shape[0]):
            term = np.dot(y[n], (np.dot(w, X[n][:]) + b))
            obj = obj + max(0, 1 - term)

        obj = self.C * obj + np.linalg.norm(w, ord=1)

        return obj


    def subgradient(self, wb, X, y):
        """Compute the subgradient of the objective function.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            subgrad (ndarray, shape = (n_features+1,)):
                subgradient of the objective function with respect to
                the coefficients wb=[w,b] of the linear model
        """

        grad_wb = np.zeros(wb.shape)
        w = wb[:-1]
        b = wb[-1]

        grad_w = np.zeros(w.shape)
        grad_b = 0.0

        for n in range(X.shape[0]):
            term = y[n]*(np.dot(w, X[n][:]) + b)
            grad_w += 0 if term > 1.0 else -np.dot(y[n], X[n][:])
        grad_w = self.C * grad_w #+ 1

        for i in range(w.shape[0]):
            if w[i] > 0:
                grad_w[i] += 1.0
            elif w[i] < 0:
                grad_w[i] -= 1.0

        for n in range(X.shape[0]):
            term = y[n]*(np.dot(w, X[n][:]) + b)
            grad_b += 0 if term > 1.0 else -y[n]
        grad_b = self.C * grad_b

        grad_wb[:-1] = np.copy(grad_w)
        grad_wb[-1] = np.copy(grad_b)

        return grad_wb


    def predict(self, X):
        """Predict class labels for samples in X.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values of -1 or 1.
        """
        [w, b] = self.get_params()

        w = w.reshape((X.shape[1],))
        y = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
             y[i] = np.dot(w, X[i][:])+b
             if y[i] >= 0:
                 y[i] = 1
             else:
                 y[i] = -1

        return y

    def get_params(self):
        """Get the model parameters.

        Returns:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        return [self.w, self.b]

    def set_params(self, w, b):
        """Set the model parameters.

        Arguments:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        self.w = w
        self.b = b

    def classification_error(self, X, y):
        [w, b] = self.get_params()

        w = w.reshape((X.shape[1],))
        y_pred = self.predict(X)

        right = 0
        wrong = 0

        for i in range(X.shape[0]):

             if y_pred[i] == y[i]:
                 right = right + 1
             else:
                 wrong = wrong + 1

        err = 1 - (right/(wrong+right))
        return err


def sparsify(X, y):

    C_axis = [1e-3, 1e-2, 1e-1, 1e0, 1e+1, 1e+2, 1e+3]
    err_axis = np.zeros(len(C_axis))
    rate_axis = np.zeros(len(C_axis))
    count  = 0

    for k in range(-3, 4):
        C = 10**k
        print('C: ',end="")
        print(C)
        clf = SVM(C, iterations=10000)
        clf.fit(X, y)
        [w, b] = clf.get_params()
        w[np.abs(w) < 1e-4] = 0
        clf.set_params(w, b)
        err_axis[count] = clf.classification_error(X, y)
        rate_axis[count] = 1 - np.count_nonzero(w)/w.shape[0]
        print('error: ',end="")
        print(err_axis[count])
        count = count + 1

    fig1 = plt.gcf()
    plt.plot(err_axis, C_axis, marker='o', linestyle='dashed', label='Observed Error Rate (X-axis) at Penalty C (Y-axis)')
    plt.yscale('log')
    plt.legend(prop={'size': 22})
    plt.xlabel('Test Error Rate', fontsize=22)
    plt.ylabel('Penalty Hyper Parameter C', fontsize=22)
    plt.title('Sparsified Model Test Error Rate with C from 1e-3 to 1e+3', fontsize=22)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    fig1.savefig('C.png')

    fig2 = plt.gcf()
    plt.plot(rate_axis, C_axis, marker='o', linestyle='dashed', label='Observed Sparisty Rate (X-axis) at Penalty C (Y-axis)')
    plt.yscale('log')
    plt.legend(prop={'size': 22})
    plt.xlabel('Sparsity Rate', fontsize=22)
    plt.ylabel('Penalty Hyper Parameter C', fontsize=22)
    plt.title('Sparsified Models Sparsity Rate with C from 1e-3 to 1e+3', fontsize=22)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    fig2.savefig('S.png')

def best(X, y):
    f1 = open('C_.txt', 'w')
    f2 = open('E_.txt', 'w')
    f3 = open('R_.txt', 'w')
    for k in np.arange(0.0005, 1, 0.0001):
        C = k
        print('C: ',end="")
        print(C,end="\t")
        f1.write("%d\n" % C)
        clf = SVM(C, iterations=10000)
        clf.fit(X, y)
        [w, b] = clf.get_params()
        w[np.abs(w) < 1e-4] = 0
        clf.set_params(w, b)
        error = clf.classification_error(X, y)
        rate = 1 - np.count_nonzero(w)/w.shape[0]
        print('error: ',end="")
        print(error,end="\t")
        f2.write("%f\n" % error)
        print('rate: ',end="")
        print(rate)
        f3.write("%f\n" % rate)

    f1.close()
    f2.close()
    f3.close()

def main():

    np.random.seed(0)

    #with gzip.open('../data/svm_data.pkl.gz', 'rb') as f:
    with gzip.open('data/svm_data.pkl.gz', 'rb') as f:
        train_X, train_y, test_X, test_y = pickle.load(f)

    #clf = SVM(C=1, iterations=10000)
    #clf.fit(train_X, train_y)
    #y_pred = clf.predict(train_X)
    #print(y_pred)
    #print(clf.classification_error(train_X, train_y))
    sparsify(test_X, test_y)
    #best(test_X, test_y)

if __name__ == '__main__':
    main()
