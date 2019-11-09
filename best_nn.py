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
#from scipy.ndimage.interpolation import map_coordinates
#from scipy.ndimage.filters import gaussian_filter

np.random.seed(1)
torch.manual_seed(1)


class BestNN():
    """A network architecture for simultaneous classification
    and angle regression of objects in images.

    Arguments:
        alpha: trade-off parameter for the composite objective function.
        epochs: number of epochs for training
    """
    def __init__(self, alpha=.5, epochs=5):
        self.alpha = alpha
        self.epochs = epochs
        #self.fc1 = nn.Linear(784, 256, bias=True)
        #self.fc2 = nn.Linear(256, 64, bias=True)
        #self.fc3 = nn.Linear(64, 32, bias=True)
        #self.fc4 = nn.Linear(64, 32, bias=True)
        #self.fc5 = nn.Linear(32, 10, bias=True)
        #self.fc6 = nn.Linear(32, 1, bias=True)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000, bias=True)
        self.fc2 = nn.Linear(1000, 10, bias=True)
        self.fc3 = nn.Linear(1000, 1, bias=True)

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

        X_t.resize_((100,1, 28,28))

        """
        # First Layer h1
        h1 = self.relu(torch.matmul(X_t, self.w1_t) + self.b1_t) + torch.cos(torch.matmul(X_t, self.w1_t) + self.b1_t)
        # Second Layer h2
        h2 = self.relu(torch.matmul(h1, self.w2_t) + self.b2_t) + torch.cos(torch.matmul(h1, self.w2_t) + self.b2_t)
        # Third Layer hp3
        h3 = self.relu(torch.matmul(h2, self.w3_t) + self.b3_t) + torch.cos(torch.matmul(h2, self.w3_t) + self.b3_t)
        # Third Layer ha3
        h4 = self.relu(torch.matmul(h2, self.w4_t) + self.b4_t) + torch.cos(torch.matmul(h2, self.w4_t) + self.b4_t)
        # Output Layer p
        y_class_pred_loss = F.cross_entropy(torch.matmul(h3, self.w5_t) + self.b5_t, y_class_t, reduction='none')
        # Output Layer a
        y_angle_pred = torch.matmul(h4, self.w6_t) + self.b6_t
        y_angle_pred = y_angle_pred.reshape((y_angle_pred.shape[0], )"""

        #h1 = F.relu(self.fc1(X_t))
        #h2 = F.relu(self.fc2(h1))
        #h3_p = F.relu(self.fc3(h2))
        #h3_a = F.relu(self.fc4(h2))
        #y_class_pred_loss = F.cross_entropy(self.fc5(h3_p), y_class_t, reduction='none')
        #y_angle_pred = self.fc6(h3_a)

        out = self.layer1(X_t)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        y_class_pred = self.fc2(out)
        y_angle_pred = self.fc3(out)

        y_class_pred_loss = F.cross_entropy(y_class_pred, y_class_t, reduction='none')

        obj = 0.0
        for n in range(X.shape[0]):
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
        y_class_f = np.zeros((1,), dtype=int)
        y_angle_f = np.zeros((1,))
        """
        h1 = self.relu(torch.matmul(X_t, self.w1_t) + self.b1_t) + torch.cos(torch.matmul(X_t, self.w1_t) + self.b1_t)
        h2 = self.relu(torch.matmul(h1, self.w2_t) + self.b2_t) + torch.cos(torch.matmul(h1, self.w2_t) + self.b2_t)
        h3 = self.relu(torch.matmul(h2, self.w3_t) + self.b3_t) + torch.cos(torch.matmul(h2, self.w3_t) + self.b3_t)
        h4 = self.relu(torch.matmul(h2, self.w4_t) + self.b4_t) + torch.cos(torch.matmul(h2, self.w4_t) + self.b4_t)
        #y_class_pred = F.cross_entropy(torch.matmul(h3, self.w5_t) + self.b5_t, y_class_t, reduction='none')
        y_class_pred = F.softmax(torch.matmul(h3, self.w5_t) + self.b5_t, dim=1)
        print(y_class_pred.shape)
        y_class_pred = y_class_pred.argmax(1)
        print(y_class_pred.shape)

        y_angle_pred = torch.matmul(h4, self.w6_t) + self.b6_t
        y_angle_pred = y_angle_pred.reshape((y_angle_pred.shape[0], ))
        """

        #h1 = F.relu(self.fc1(X_t))
        #h2 = F.relu(self.fc2(h1))
        #h3_p = F.relu(self.fc3(h2))
        #h3_a = F.relu(self.fc4(h2))
        #y_class_pred = F.softmax(self.fc5(h3_p), dim=1)

        for i in range(0, X_t.shape[0], 100):
            X_t = X_t[i:i+100-1][:]
            X_t.resize_((100,1, 28,28))
            out = self.layer1(X_t)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.drop_out(out)
            out = self.fc1(out)
            y_class_pred = self.fc2(out)
            y_class_pred = y_class_pred.argmax(1)
            #y_angle_pred = self.fc6(h3_a)
            y_angle_pred = self.fc3(out)
            y_angle_pred = y_angle_pred.reshape((y_angle_pred.shape[0], ))

            y_class_pred = y_class_pred.data.numpy()
            y_angle_pred = y_angle_pred.data.numpy()

            y_class_f = np.append(y_class_f, y_class_pred)
            y_angle_f = np.append(y_angle_f, y_angle_pred)

        print(y_class_f[1:].shape)
        print(y_angle_f[1:].shape)

        return [y_class_f[1:], y_angle_f[1:]]

    def fit(self, X, y_class, y_angle, step=1e-4):
        """Train the model according to the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 784)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Labels of objects. Each entry is in [0,...,C-1].
            y_angle (numpy ndarray, shape = (samples, )):
                Angles of the objects in degrees.
        """

        optimizer = torch.optim.Adam([self.layer1[0].weight, self.layer1[0].bias, self.layer2[0].weight, self.layer2[0].bias, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias, self.fc3.weight, self.fc3.bias], lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)

        n_epochs = self.epochs
        #n_epochs = 6
        batch_size = 100

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

    def calc_error(self, X_tr, y_tr, a_tr, X_val, y_val, a_val):
        [y_tr_pred, a_tr_pred] = self.predict(X_tr)
        [y_val_pred, a_val_pred] = self.predict(X_val)

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
        class_err_2 = true2/y_val_pred.shape[0]

        mae1 = mae1/y_tr_pred.shape[0]
        mae2 = mae2/y_val_pred.shape[0]

        print('Classification Accuracy on Train: ', end='')
        print(class_err_1)
        print('Classification Accuracy on Valid: ', end='')
        print(class_err_2)
        print('Mean Absolute on Train: ', end='')
        print(mae1)
        print('Mean Absolute on Valid: ', end='')
        print(mae2)

        #Classification Accuracy on Train: 0.98
        #Classification Accuracy on Valid: 0.96
        #Mean Absolute on Train: 10.528078113113327
        #Mean Absolute on Valid: 10.505664949393335

        return [class_err_1, class_err_2, mae1, mae2]

    def preprocess(self, X_tr, X_val, X_te):

        random_state = np.random.RandomState(None)
        shape = (28,28)
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), 4, mode="constant", cval=0) * 34
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), 4, mode="constant", cval=0) * 34
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

        for n in range(X_tr.shape[0]):
            img = (X_tr[n] * 256).reshape((28, 28))
            X_tr[n] = map_coordinates(img, indices, order=1).reshape(shape).flatten()

        for n in range(X_val.shape[0]):
            img = (X_val[n] * 256).reshape((28, 28))
            X_val[n] = map_coordinates(img, indices, order=1).reshape(shape).flatten()

        for n in range(X_te.shape[0]):
            img = (X_te[n] * 256).reshape((28, 28))
            X_te[n] = map_coordinates(img, indices, order=1).reshape(shape).flatten()

        return [X_tr, X_val, X_te]

    def savetestpred(self, X_te):
        [y_te_pred, a_te_pred] = self.predict(X_te)
        print(y_te_pred.shape)
        print(a_te_pred.shape)
        np.save("code/class_predictions_alpha1.npy", y_te_pred, allow_pickle=True)
        np.save("code/angle_predictions_alpha1.npy", a_te_pred, allow_pickle=True)

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

    #plt.imshow(X_tr[0].reshape((28,28)))
    #plt.show()
    nn = BestNN(1.0, 30)
    #[X_tr, X_val, X_te] = nn.preprocess(X_tr, X_val, X_te)

    nn.fit(X_tr, y_tr, a_tr)
    nn.predict(X_tr)
    nn.calc_error(X_tr, y_tr, a_tr, X_val, y_val, a_val)
    nn.savetestpred(X_te)

if __name__ == '__main__':
    main()
