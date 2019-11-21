import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
import os
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

class mixture_model:
    """A Laplace mixture model trained using marginal likelihood maximization

    Arguments:
        K: number of mixture components

    """
    def __init__(self, K=5):
        self.K = K
        self.batch_size = 100
        self.constlogsumexp = 5.0

    def get_params(self):
        """Get the model parameters.

        Returns:
            a list containing the following parameter values:

            mu (numpy ndarray, shape = (D, K))
            b (numpy ndarray, shape = (D,K))
            pi (numpy ndarray, shape = (K,))

        """
        return [self.mu_t.data.numpy(), np.exp(self.b_new_t.data.numpy()), np.exp(self.pi_new_t.data.numpy() - self.constlogsumexp)]

    def set_params(self, mu, b, pi):
        """Set the model parameters.

        Arguments:
            mu (numpy ndarray, shape = (D, K))
            b (numpy ndarray, shape = (D,K))
            pi (numpy ndarray, shape = (K,))
        """
        self.mu = mu
        self.b = b
        self.pi = pi

        self.mu_t = torch.tensor(self.mu, requires_grad=True)
        self.b_t = torch.tensor(self.b, requires_grad=True)
        self.pi_t = torch.tensor(self.pi, requires_grad=True)

        self.b_new = np.log(self.b)
        self.pi_new = np.log(self.pi) + self.constlogsumexp
        self.b_new_t = torch.tensor(self.b_new, requires_grad=True)
        self.pi_new_t = torch.tensor(self.pi_new, requires_grad=True)

    def marginal_likelihood(self, X):
        """log marginal likelihood function.
           Computed using the current values of the parameters
           as set via fit or set_params.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's

        Returns:
            Marginal likelihood of observed data
        """
        [N, D] = X.shape
        X_t = torch.from_numpy(X).float()

        sum_n = torch.tensor(0, dtype=torch.float)
        prod = torch.zeros(self.K, dtype=torch.float)
        for n in range(N):
            for z in range(self.K):
                term = torch.tensor(0, dtype=torch.float)
                for d in range(D):
                    if not(torch.isnan(X_t[n][d])):
                        term = term - torch.log(torch.tensor(2, dtype=torch.float)) - self.b_new_t[d][z] - (torch.abs(X_t[n][d] - self.mu_t[d][z])/torch.exp(self.b_new_t[d][z]))
                prod[z] = self.pi_new_t[z] - torch.logsumexp(self.pi_new_t, dim=0) + term
            sum_n = sum_n + torch.logsumexp(prod.clone(), dim=0)

        print(sum_n)
        self.marginal = sum_n
        return sum_n.data.numpy()

    def predict_proba(self, X):
        """Predict the probability over clusters P(Z=z|X=x) for each example in X.
           Use the currently stored parameter values.

        Arguments:
            X (numpy ndarray, shape = (N,D)):
                Input matrix where each row is a feature vector.

        Returns:
            PZ (numpy ndarray, shape = (N,K)):
                Probability distribution over classes for each
                data case in the data set.
        """
        [N, D] = X.shape
        X_t = torch.from_numpy(X).float()
        PZ = torch.zeros([N, self.K], dtype=torch.float)

        for n in range(N):
            sum_z = torch.tensor(0, dtype=torch.float)
            for z in range(self.K):
                term = torch.tensor(0, dtype=torch.float)
                for d in range(D):
                    if not(torch.isnan(X_t[n][d])):
                        term = term - torch.log(torch.tensor(2, dtype=torch.float)) - self.b_new_t[d][z] - (torch.abs(X_t[n][d] - self.mu_t[d][z])/torch.exp(self.b_new_t[d][z]))
                prod = torch.exp(self.pi_new_t[z] - torch.logsumexp(self.pi_new_t, dim=0) + term)
                PZ[n][z] = prod
                sum_z = sum_z + prod
            PZ[n][:] = PZ[n][:] / sum_z

        return PZ.data.numpy()

    def impute(self, X):
        """Mean imputation of missing values in the input data matrix X.
           Ipmute based on the currently stored parameter values.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's

        Returns:
            XI (numpy ndarray, shape = (N, D)):
                The input data matrix where the missing values on
                each row (indicated by np.nans) have been imputed
                using their conditional means given the observed
                values on each row.
        """
        #X = X[:500]
        [N, D] = X.shape
        X_t = torch.from_numpy(X).float()
        X_new_t = X_t
        PZ = self.predict_proba(X)
        PZ_t = torch.from_numpy(PZ).float()

        #print(PZ)
        #print(tuple_class_t)
        #print(X_t)
        for n in range(N):
            for d in range(D):
                if torch.isnan(X_t[n][d]):
                    X_t[n][d] = torch.sum(self.mu_t[d, :]*PZ_t[n, :])

        print(X_t)
        #print(Xi)
        return X_new_t.data.numpy()

    def transform(self, X):
        M  = 0 + np.isnan(X)
        X2 = np.copy(X)
        X2[M==1] = 0
        N,D  = X2.shape
        X_t = torch.from_numpy(X2).unsqueeze(2)
        X_new_t = 1-torch.from_numpy(M).unsqueeze(2)
        return X_t, X_new_t

    def marginal_likelihood_1(self, X, M, mu, b_new, pi_new):
        log_pi_normed = F.log_softmax(pi_new,dim=1)
        scaled_dist = torch.abs(X - mu)/torch.exp(b_new)
        log_norm = -torch.log(torch.tensor([2.0])) - b_new
        ll = log_pi_normed
        ll = ll + torch.sum(torch.mul(M, log_norm),dim=1)
        ll = ll - torch.sum(torch.mul(M, scaled_dist),dim=1)
        mll = torch.sum(torch.logsumexp(ll,1))
        return -1*mll

    def fit(self, X, mu_init=None, b_init=None, pi_init=None, step=0.1, epochs=20):
        """Train the model according to the given training data
           by directly maximizing the marginal likelihood of
           the observed data. If initial parameters are specified, use those
           to initialize the model. Otherwise, use a random initialization.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's
            mu_init (None or numpy ndarray, shape = (D, K)):
                Array of Laplace density mean paramaeters for each mixture component
                to use for initialization
            b_init (None or numpy ndarray, shape = (D, K)):
                Array of Laplace density scale parameters for each mixture component
                to use for initialization
            pi_init (None or numpy ndarray, shape = (K,)):
                Mixture proportions to use for initialization
            step (float):
                Initial step size to use during training
            epochs (int): number of epochs for training
        """
        N,D = X.shape
        #self.set_params((torch.nn.Parameter(torch.randn(X.shape[1], self.K), requires_grad=True)).data.numpy(), torch.rand((X.shape[1], self.K), requires_grad=True).data.numpy(), np.repeat(1/self.K, self.K))
        self.mu_t      = torch.randn([1,D,self.K], requires_grad=True)
        self.b_new_t   = torch.zeros([1,D,self.K], requires_grad=True)
        self.pi_new_t  = torch.zeros([1,self.K], requires_grad=True)
        print(self.mu_t.shape)
        X_t,X_new_t = self.transform(X)
        data  = TensorDataset(X_t, X_new_t)
        data_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=self.batch_size)

        #torch.autograd.set_detect_anomaly(True)
        #print(self.b_new_t.tolist())
        optimizer = torch.optim.Adam([self.mu_t, self.b_new_t, self.pi_new_t], lr=step)
        #optimizer = torch.optim.SGD([self.mu_t, self.b_new_t, self.pi_new_t], lr=step, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=False)

        for epoch in range(epochs):
            for batch_x, batch_m in data_loader:
                optimizer.zero_grad()
                loss = self.marginal_likelihood_1(batch_x,batch_m, self.mu_t,self.b_new_t,self.pi_new_t)
                #print(loss)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    loss = self.marginal_likelihood_1(X_t,X_new_t,self.mu_t,self.b_new_t,self.pi_new_t)
                    mu,b,pi = self.get_params()
            print(loss)
        #print(self.mu_t.squeeze().shape)
        self.mu_t = self.mu_t.squeeze()
        self.b_new_t = self.b_new_t.squeeze()
        self.pi_new_t = self.pi_new_t.squeeze()
        print(self.get_params())

        #n_epochs = self.epochs
        """n_epochs = epochs
        losses = np.zeros((n_epochs,1),dtype=float)
        #batch_size = 100

        for epoch in range(n_epochs):
            #print(epoch)
            #permutation = np.random.permutation(X.shape[0])
            #X = X[permutation]

            #for i in range(0, X.shape[0], batch_size):
            #    print(i)
            optimizer.zero_grad()       # clear gradients for next train

            #    if (i + batch_size <= X.shape[0]):
            #        indices = permutation[i:i+batch_size]
            #    else:
            #        indices = permutation[i:X.shape[0]]
            #    batch_x = X[indices]

            #loss = -torch.tensor(self.marginal_likelihood(X), requires_grad=True)
            self.marginal_likelihood(X)
            loss = -self.marginal
            losses[epoch] = self.marginal.data.numpy()
            #loss = -self.marginal_likelihood(X)
            #print(loss)
            loss.backward()             # backpropagation, compute gradients
            optimizer.step()            # apply gradients"""

    def plot(self, X1, X2, step=0.1, epochs=20):
        """Train the model according to the given training data
           by directly maximizing the marginal likelihood of
           the observed data. If initial parameters are specified, use those
           to initialize the model. Otherwise, use a random initialization.

        Arguments:
            X (numpy ndarray, shape = (N, D)):
                Input matrix where each row is a feature vector.
                Missing data is indicated by np.nan's
            mu_init (None or numpy ndarray, shape = (D, K)):
                Array of Laplace density mean paramaeters for each mixture component
                to use for initialization
            b_init (None or numpy ndarray, shape = (D, K)):
                Array of Laplace density scale parameters for each mixture component
                to use for initialization
            pi_init (None or numpy ndarray, shape = (K,)):
                Mixture proportions to use for initialization
            step (float):
                Initial step size to use during training
            epochs (int): number of epochs for training
        """

        marg_K1 = torch.zeros([20, 1], dtype=torch.float)
        marg_K2 = torch.zeros([20, 1], dtype=torch.float)
        n_epochs = epochs

        for K in range(1, 21):
            print(K)
            self.K = K
            N,D = X1.shape
            #self.set_params((torch.nn.Parameter(torch.randn(X.shape[1], self.K), requires_grad=True)).data.numpy(), torch.rand((X.shape[1], self.K), requires_grad=True).data.numpy(), np.repeat(1/self.K, self.K))
            self.mu_t      = torch.randn([1,D,self.K], requires_grad=True)
            self.b_new_t   = torch.zeros([1,D,self.K], requires_grad=True)
            self.pi_new_t  = torch.zeros([1,self.K], requires_grad=True)
            X_t,X_new_t = self.transform(X1)
            data  = TensorDataset(X_t, X_new_t)
            data_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=self.batch_size)
            optimizer = torch.optim.Adam([self.mu_t, self.b_new_t, self.pi_new_t], lr=step)
            #optimizer = torch.optim.SGD([self.mu_t, self.b_new_t, self.pi_new_t], lr=step, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=False)

            for epoch in range(epochs):
                for batch_x, batch_m in data_loader:
                    optimizer.zero_grad()
                    loss = self.marginal_likelihood_1(batch_x,batch_m, self.mu_t,self.b_new_t,self.pi_new_t)
                    #print(loss)
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        loss = self.marginal_likelihood_1(X_t,X_new_t,self.mu_t,self.b_new_t,self.pi_new_t)
                        mu,b,pi = self.get_params()
                print(loss)
            marg_K1[K-1] = loss

        for K in range(1, 21):
            print(K)
            self.K = K
            N,D = X2.shape
            #self.set_params((torch.nn.Parameter(torch.randn(X.shape[1], self.K), requires_grad=True)).data.numpy(), torch.rand((X.shape[1], self.K), requires_grad=True).data.numpy(), np.repeat(1/self.K, self.K))
            self.mu_t      = torch.randn([1,D,self.K], requires_grad=True)
            self.b_new_t   = torch.zeros([1,D,self.K], requires_grad=True)
            self.pi_new_t  = torch.zeros([1,self.K], requires_grad=True)
            X_t,X_new_t = self.transform(X2)
            data  = TensorDataset(X_t, X_new_t)
            data_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=self.batch_size)
            optimizer = torch.optim.Adam([self.mu_t, self.b_new_t, self.pi_new_t], lr=step)
            #optimizer = torch.optim.SGD([self.mu_t, self.b_new_t, self.pi_new_t], lr=step, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=False)

            for epoch in range(epochs):
                for batch_x, batch_m in data_loader:
                    optimizer.zero_grad()
                    loss = self.marginal_likelihood_1(batch_x,batch_m, self.mu_t,self.b_new_t,self.pi_new_t)
                    #print(loss)
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        loss = self.marginal_likelihood_1(X_t,X_new_t,self.mu_t,self.b_new_t,self.pi_new_t)
                        mu,b,pi = self.get_params()
                print(loss)
            marg_K2[K-1] = loss


        fig1 = plt.gcf()
        plt.plot(np.arange(1,21), marg_K1.data.numpy(), marker='o', linestyle='dashed', label='TR1')
        plt.plot(np.arange(1,21), marg_K2.data.numpy(), marker='o', linestyle='dashed', label='TE1')
        #plt.plot(np.arange(1,21), [-73887.0469, -21449.4277, -17628.6074, -18599.6465, -17044.7988, -18936.5137, -16239.6768, -16141.1816, -16018.0674, -16000.3975, -15985.5703, -15988.5029, -15852.6387, -16013.0156, -15810.3340, -15775.1768,-15890.4932,-15968.7178,-15963.0830, -16180.8447], marker='o', linestyle='dashed', label='TR1')
        #plt.plot(np.arange(1,21), [-37184.4297, -10747.1729, -8836.6816, -9359.1660, -8624.4326, -9487.2129, -8235.6279, -8133.8994, -8042.6094, -8036.4160, -8000.7100, -7965.3853, -7971.5088, -7961.9048, -7950.3359, -7912.3018, -7877.2251, -7852.1211, -7860.0640, -8043.0981], marker='o', linestyle='dashed', label='TE1')
        plt.legend(prop={'size': 22})
        plt.xlabel('Number of Class Labels K', fontsize=22)
        plt.xticks(np.arange(1, 21, step=1))
        plt.ylabel('Negative Log Marginal Likelihood', fontsize=22)
        plt.title(r'Negative Log Marginal Likelihood(Y-axis) for Class Labels K $\in$ {1,...,20}(X-axis) at 20 epochs', fontsize=22)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
        fig1.savefig('q2f_2.png')

    def imputation_error(self, X1, X2, step=0.1, epochs=20):
        error = torch.zeros([20, 1], dtype=torch.float)
        n_epochs = epochs

        for K in range(1, 21):
            count = 0
            print(K)
            self.K = K
            N,D = X1.shape
            #self.set_params((torch.nn.Parameter(torch.randn(X.shape[1], self.K), requires_grad=True)).data.numpy(), torch.rand((X.shape[1], self.K), requires_grad=True).data.numpy(), np.repeat(1/self.K, self.K))
            self.mu_t      = torch.randn([1,D,self.K], requires_grad=True)
            self.b_new_t   = torch.zeros([1,D,self.K], requires_grad=True)
            self.pi_new_t  = torch.zeros([1,self.K], requires_grad=True)
            X_t,X_new_t = self.transform(X1)
            data  = TensorDataset(X_t, X_new_t)
            data_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=self.batch_size)
            optimizer = torch.optim.Adam([self.mu_t, self.b_new_t, self.pi_new_t], lr=step)
            #optimizer = torch.optim.SGD([self.mu_t, self.b_new_t, self.pi_new_t], lr=step, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=False)

            for epoch in range(epochs):
                for batch_x, batch_m in data_loader:
                    optimizer.zero_grad()
                    loss = self.marginal_likelihood_1(batch_x,batch_m, self.mu_t,self.b_new_t,self.pi_new_t)
                    #print(loss)
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        loss = self.marginal_likelihood_1(X_t,X_new_t,self.mu_t,self.b_new_t,self.pi_new_t)
                        mu,b,pi = self.get_params()
                print(loss)
            self.mu_t = self.mu_t.squeeze()
            self.b_new_t = self.b_new_t.squeeze()
            self.pi_new_t = self.pi_new_t.squeeze()

            if K == 1:
                self.mu_t = self.mu_t.reshape(D, 1)
                self.b_new_t = self.b_new_t.reshape(D, 1)
                self.pi_new_t = self.pi_new_t.reshape(K, 1)

            X1_im = self.impute(X1)
            for n in range(N):
                for d in range(D):
                    if ~np.isnan(X2[n][d]):
                        error[K-1] += np.abs(X1_im[n][d] - X2[n][d])
                        count += 1
            error[K-1] /= count

        fig1 = plt.gcf()
        plt.plot(np.arange(1,21), error.data.numpy(), marker='o', linestyle='dashed', label='MAE when model is fit on K classes')
        #plt.plot(np.arange(1,21), [-73887.0469, -21449.4277, -17628.6074, -18599.6465, -17044.7988, -18936.5137, -16239.6768, -16141.1816, -16018.0674, -16000.3975, -15985.5703, -15988.5029, -15852.6387, -16013.0156, -15810.3340, -15775.1768,-15890.4932,-15968.7178,-15963.0830, -16180.8447], marker='o', linestyle='dashed', label='TR1')
        #plt.plot(np.arange(1,21), [-37184.4297, -10747.1729, -8836.6816, -9359.1660, -8624.4326, -9487.2129, -8235.6279, -8133.8994, -8042.6094, -8036.4160, -8000.7100, -7965.3853, -7971.5088, -7961.9048, -7950.3359, -7912.3018, -7877.2251, -7852.1211, -7860.0640, -8043.0981], marker='o', linestyle='dashed', label='TE1')
        plt.legend(prop={'size': 22})
        plt.xlabel('Number of Class Labels K', fontsize=22)
        plt.xticks(np.arange(1, 21, step=1))
        plt.ylabel('Mean Absolute Imputation Error\nbetween imputed TR2 and TE2', fontsize=22)
        plt.title(r'Mean Absolute Imputation Error(Y-axis) for Class Labels K $\in$ {1,...,20}(X-axis) at 20 epochs', fontsize=22)
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()
        fig1.savefig('q3e.png')

def main():

    data=np.load("../data/data.npz")
    xtr1 = data["xtr1"]
    xtr2 = data["xtr2"]
    xte1 = data["xte1"]
    xte2 = data["xte2"]

    mm = mixture_model(K=5)
    #mm.fit(xtr2)
    #xtr2_im = mm.impute(xtr2)
    #print(xtr2)
    #print(xtr2_im)
    #print(xte2)
    #mm.imputation_error(xtr2, xte2)
    mm.plot(xtr1, xte1)

if __name__ == '__main__':
    main()
