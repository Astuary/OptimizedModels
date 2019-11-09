import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
import gzip
import pickle
import os
#import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)



class NN:
    """A network architecture for simultaneous classification
    and angle regression of objects in images.

    Arguments:
        alpha: trade-off parameter for the composite objective function.
        epochs: number of epochs for training
    """
    def __init__(self, alpha=.5, epochs=5):
        self.alpha = alpha
        self.epochs = epochs
        self.fc1 = nn.Linear(784, 256, bias=True)
        self.fc2 = nn.Linear(256, 64, bias=True)
        self.fc3 = nn.Linear(64, 32, bias=True)
        self.fc4 = nn.Linear(64, 32, bias=True)
        self.fc5 = nn.Linear(32, 10, bias=True)
        self.fc6 = nn.Linear(32, 1, bias=True)

    def objective(self, X, y_class, y_angle):
        """Objective function.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Labels of objects. Each entry is in [0,...,C-1].
            y_angle (numpy ndarray, shape = (samples, )):
                Angles of the objects in degrees.

        Returns:
            Composite objective function value.
        """
        X_t = torch.from_numpy(X).float()
        y_class_t = torch.from_numpy(y_class)
        y_angle_t = torch.from_numpy(y_angle).float()

        h1 = F.relu(self.fc1(X_t))
        h2 = F.relu(self.fc2(h1))
        h3_p = F.relu(self.fc3(h2))
        h3_a = F.relu(self.fc4(h2))
        y_class_pred_loss = F.cross_entropy(self.fc5(h3_p), y_class_t, reduction='none')
        y_angle_pred = self.fc6(h3_a)

        obj = 0.0
        for n in range(X.shape[0]):
            #print(obj)
            obj = obj + self.alpha*(y_class_pred_loss[n]) + (1 - self.alpha)*0.5*(1 - torch.cos(0.01745*(y_angle[n] - y_angle_pred[n])))

        return obj.data.numpy()     #For Autograder
        #return obj               #For local

    def predict(self, X):
        """Predict class labels and object angles for samples in X.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Input matrix where each row is a feature vector.

        Returns:
            y_class (numpy ndarray, shape = (samples,)):
                predicted labels. Each entry is in [0,...,C-1].
            y_angle (numpy ndarray, shape = (samples, )):
                The predicted angles of the imput objects.
        """
        X_t = torch.from_numpy(X).float()

        h1 = F.relu(self.fc1(X_t))
        h2 = F.relu(self.fc2(h1))
        h3_p = F.relu(self.fc3(h2))
        h3_a = F.relu(self.fc4(h2))
        y_class_pred = F.softmax(self.fc5(h3_p), dim=1)
        y_class_pred = y_class_pred.argmax(1)
        y_angle_pred = self.fc6(h3_a)
        y_angle_pred = y_angle_pred.reshape((y_angle_pred.shape[0], ))

        y_class_pred = y_class_pred.data.numpy()
        y_angle_pred = y_angle_pred.data.numpy()

        #print(y_class_pred.shape)
        #print(y_angle_pred.shape)

        return [y_class_pred, y_angle_pred]

    def fit(self, X, y_class, y_angle ,step=1e-4):
        """Train the model according to the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Labels of objects. Each entry is in [0,...,C-1].
            y_angle (numpy ndarray, shape = (samples, )):
                Angles of the objects in degrees.
        """
        optimizer = torch.optim.Adam([self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias, self.fc3.weight, self.fc3.bias, self.fc4.weight, self.fc4.bias, self.fc5.weight, self.fc5.bias, self.fc6.weight, self.fc6.bias], lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)

        n_epochs = self.epochs
        #n_epochs = 1
        batch_size = 64
        print(self.alpha)
        for epoch in range(n_epochs):
            print(epoch)
            permutation = np.random.permutation(X.shape[0])
            X = X[permutation]
            y_class = y_class[permutation]
            y_angle = y_angle[permutation]

            for i in range(0, X.shape[0], batch_size):
                #print(i)
                optimizer.zero_grad()       # clear gradients for next train

                if (i + batch_size <= X.shape[0]):
                    indices = permutation[i:i+batch_size]
                else:
                    indices = permutation[i:X.shape[0]]
                batch_x, batch_y_class, batch_y_angle = X[indices], y_class[indices], y_angle[indices]

                loss = self.objective(batch_x, batch_y_class, batch_y_angle)
                loss.backward()             # backpropagation, compute gradients
                optimizer.step()            # apply gradients

    def get_params(self):
        """Get the model parameters.

        Returns:
            a list containing the following parameter values represented
            as numpy arrays (see handout for definitions of each parameter).

            w1 (numpy ndarray, shape = (784, 256))
            b1 (numpy ndarray, shape = (256,))
            w2 (numpy ndarray, shape = (256, 64))
            b2 (numpy ndarray, shape = (64,))
            w3 (numpy ndarray, shape = (64, 32))
            b3 (numpy ndarray, shape = (32,))
            w4 (numpy ndarray, shape = (64, 32))
            b4 (numpy ndarray, shape = (32,))
            w5 (numpy ndarray, shape = (32, 10))
            b5 (numpy ndarray, shape = (10,))
            w6 (numpy ndarray, shape = (32, 1))
            b6 (numpy ndarray, shape = (1,))
        """
        #return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5, self.b5, self.w6, self.b6]
        print(np.transpose(self.fc1.bias.data.numpy()).shape)
        return [np.transpose(self.fc1.weight.data.numpy()), np.transpose(self.fc1.bias.data.numpy()), np.transpose(self.fc2.weight.data.numpy()), np.transpose(self.fc2.bias.data.numpy()), np.transpose(self.fc3.weight.data.numpy()), np.transpose(self.fc3.bias.data.numpy()), np.transpose(self.fc4.weight.data.numpy()), np.transpose(self.fc4.bias.data.numpy()), np.transpose(self.fc5.weight.data.numpy()), np.transpose(self.fc5.bias.data.numpy()), np.transpose(self.fc6.weight.data.numpy()), np.transpose(self.fc6.bias.data.numpy())]

    def set_params(self, params):
        """Set the model parameters.

        Arguments:
            params is a list containing the following parameter values represented
            as numpy arrays (see handout for definitions of each parameter).

            w1 (numpy ndarray, shape = (784, 256))
            b1 (numpy ndarray, shape = (256,))
            w2 (numpy ndarray, shape = (256, 64))
            b2 (numpy ndarray, shape = (64,))
            w3 (numpy ndarray, shape = (64, 32))
            b3 (numpy ndarray, shape = (32,))
            w4 (numpy ndarray, shape = (64, 32))
            b4 (numpy ndarray, shape = (32,))
            w5 (numpy ndarray, shape = (32, 10))
            b5 (numpy ndarray, shape = (10,))
            w6 (numpy ndarray, shape = (32, 1))
            b6 (numpy ndarray, shape = (1,))
        """
        self.fc1.weight = nn.Parameter(torch.from_numpy(np.transpose(params[0])))
        self.fc1.bias = nn.Parameter(torch.from_numpy(np.transpose(params[1])))
        self.fc2.weight = nn.Parameter(torch.from_numpy(np.transpose(params[2])))
        self.fc2.bias = nn.Parameter(torch.from_numpy(np.transpose(params[3])))
        self.fc3.weight = nn.Parameter(torch.from_numpy(np.transpose(params[4])))
        self.fc3.bias = nn.Parameter(torch.from_numpy(np.transpose(params[5])))
        self.fc4.weight = nn.Parameter(torch.from_numpy(np.transpose(params[6])))
        self.fc4.bias = nn.Parameter(torch.from_numpy(np.transpose(params[7])))
        self.fc5.weight = nn.Parameter(torch.from_numpy(np.transpose(params[8])))
        self.fc5.bias = nn.Parameter(torch.from_numpy(np.transpose(params[9])))
        self.fc6.weight = nn.Parameter(torch.from_numpy(np.transpose(params[10])))
        self.fc6.bias = nn.Parameter(torch.from_numpy(np.transpose(params[11])))

    def calc_error(self, X_tr, y_tr, a_tr, X_val, y_val, a_val):
        [y_tr_pred, a_tr_pred] = self.predict(X_tr)
        [y_val_pred, a_val_pred] = self.predict(X_val)

        #a_tr = torch.from_numpy(a_tr)
        #a_val = torch.from_numpy(a_val)
        #a_tr_pred = a_tr_pred.numpy()
        #a_val_pred = a_val_pred.numpy()

        """
        print('class train: ')
        print(y_tr)
        print(type(y_tr))
        print('angle train: ')
        print(a_tr)
        print(type(a_tr))

        print('class train predict: ')
        print(y_tr_pred)
        print(type(y_tr_pred))
        print('angle train predict: ')
        print(a_tr_pred)
        print(type(a_tr_pred))

        print('class valid: ')
        print(y_val)
        print(type(y_val))
        print('angle valid: ')
        print(a_val)
        print(type(a_val))

        print('class valid predict: ')
        print(y_val_pred)
        print(type(y_val_pred))
        print('angle valid: ')
        print(a_val_pred)
        print(type(a_val_pred))
        """

        true1 = 0
        true2 = 0
        mae1 = 0.0
        mae2 = 0.0

        for i in range(y_tr_pred.shape[0]):
            if y_tr_pred[i] == y_tr[i]:
                true1 = true1 + 1
            mae1 += np.abs(a_tr_pred[i] - a_tr[i])

        for i in range(y_val_pred.shape[0]):
            if y_val_pred[i] == y_val[i]:
                true2 = true2 + 1
            mae2 += np.abs(a_val_pred[i] - a_val[i])

        class_err_1 = true1/y_tr_pred.shape[0]
        class_err_1 = 1 - class_err_1
        class_err_2 = true2/y_val_pred.shape[0]
        class_err_2 = 1 - class_err_2

        mae1 = mae1/y_tr_pred.shape[0]
        mae2 = mae2/y_val_pred.shape[0]

        print('Classification Accuracy on Train: ', end='')
        print(class_err_1)
        print('Classification Accuracy on Valid: ', end='')
        print(class_err_2)
        print('Mean Absolute Error on Train: ', end='')
        print(mae1)
        print('Mean Absolute Error on Valid: ', end='')
        print(mae2)
        return [class_err_1, class_err_2, mae1, mae2]


def plot(X_tr, y_tr, a_tr, X_val, y_val, a_val):
    class_err = np.zeros((11,1))
    mae = np.zeros((11,1))
    count = 0

    for i in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        nn = NN(i, 20)
        nn.fit(X_tr, y_tr, a_tr)
        nn.predict(X_tr)
        [class_err_1, class_err_2, mae1, mae2] = nn.calc_error(X_tr, y_tr, a_tr, X_val, y_val, a_val)
        class_err[count] = class_err_2
        mae[count] = mae2
        count = count + 1

    fig1 = plt.gcf()
    plt.plot(np.arange(0.0, 1.1, 0.1).tolist(), class_err, marker='o', linestyle='dashed', label=r'Observed Classification Error Rate (Y-axis) at Trade-off parameter $\alpha$ (X-axis)')
    plt.legend(prop={'size': 22})
    plt.ylabel('Classification Error on Validation Data', fontsize=22)
    plt.yticks(np.arange(0.0, 1.1, 0.1).tolist())
    plt.xlabel(r'Trade-off parameter $\alpha$', fontsize=22)
    plt.xticks(np.arange(0.0, 1.1, 0.1).tolist())
    plt.title('Classification Error vs Trade-off Parameter', fontsize=22)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    fig1.savefig('C.png')

    fig2 = plt.gcf()
    plt.plot(np.arange(0.0, 1.1, 0.1).tolist(), mae, marker='o', linestyle='dashed', label=r'Observed Mean Absolute Error (Y-axis) at Trade-off parameter $\alpha$ (X-axis)')
    plt.legend(prop={'size': 22})
    plt.ylabel('Mean Absolute Error on Validation Data', fontsize=22)
    plt.xlabel(r'Trade-off parameter $\alpha$', fontsize=22)
    plt.xticks(np.arange(0.0, 1.1, 0.1).tolist())
    plt.title('Mean Absolute Error vs Trade-off Parameter', fontsize=22)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    fig2.savefig('M.png')

def main():

    DATA_DIR = 'data'
    data=np.load(os.path.join(DATA_DIR, "mnist_rot_train.npz"))
    X_tr,y_tr,a_tr = data["X"],data["labels"],data["angles"]

    data=np.load(os.path.join(DATA_DIR, "mnist_rot_validation.npz"))
    X_val,y_val,a_val = data["X"],data["labels"],data["angles"]

    #Note: test class labels and angles are not provided
    #in the data set
    data=np.load(os.path.join(DATA_DIR, "mnist_rot_test.npz"))
    X_te,y_te,a_te = data["X"],data["labels"],data["angles"]

    #nn = NN(0.5, 20)
    #nn.fit(X_tr, y_tr, a_tr)
    #nn.predict(X_tr)
    #nn.calc_error(X_tr, y_tr, a_tr, X_val, y_val, a_val)

    plot(X_tr, y_tr, a_tr, X_val, y_val, a_val)


if __name__ == '__main__':
    main()
