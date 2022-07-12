import numpy as np
import matplotlib.pyplot as plt


np.random.seed(1)
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] <= 0.5:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def derivative_mse(y, pred_y):
    return -2 * (y - pred_y) / y.shape[0]

def loss_mse(y, pred_y):
    return np.mean((y - pred_y)**2)

def ReLU(x):
    x[x<0] = 0
    return x

def derivative_ReLU(x):
    index_1 = x>=0
    index_2 = x<0
    x[index_1] = 1
    x[index_2] = 0
    return x
    # return 1 if x >= 0 else 0

class MLP():

    def __init__(self, units1=10, units2=10, units3=1):
        self.w1 = np.random.normal(0, 1, (2, units1))
        self.w2 = np.random.normal(0, 1, (units1, units2))
        self.w3 = np.random.normal(0, 1, (units2, units3))
        self.w1_sum = 0
        self.w2_sum = 0
        self.w3_sum = 0

    def forward(self, x):
        self.x = x
        self.a1 = x @ self.w1
        self.z1 = sigmoid(self.a1)
        self.a2 = self.z1 @ self.w2
        self.z2 = sigmoid(self.a2)
        self.a3 = self.z2 @ self.w3
        self.pred_y = sigmoid(self.a3)

        return self.pred_y

    def backward(self, y):
        '''
        backward propagation
        '''
        dl_dy = derivative_mse(y, self.pred_y)
        dy_da3 = derivative_sigmoid(self.pred_y)
        da3_dw3 = self.z2
        G1 = dy_da3 * dl_dy
        self.dl_dw3 = da3_dw3.T @ G1

        da3_dz2 = self.w3
        dz2_da2 = derivative_sigmoid(self.z2)
        da2_dw2 = self.z1
        G2 = dz2_da2 * (G1 @ self.w3.T)
        self.dl_dw2 = da2_dw2.T @ G2

        da2_dz1 = self.w2
        dz1_da1 = derivative_sigmoid(self.z1)
        da1_dw1 = self.x
        G3 = dz1_da1 * (G2 @ self.w2.T)
        self.dl_dw1 = da1_dw1.T @ G3

    def update(self, lr, optimizer=''):
        '''
        Update weight
        '''
        self.lr = lr
        if optimizer == 'adagrad':
            self.w1_sum += (self.w1**2)
            self.w2_sum += (self.w2**2)
            self.w3_sum += (self.w3**2)
            self.w1 = self.w1 - self.lr * self.dl_dw1 / (self.w1_sum**0.5)
            self.w2 = self.w2 - self.lr * self.dl_dw2 / (self.w2_sum**0.5)
            self.w3 = self.w3 - self.lr * self.dl_dw3 / (self.w3_sum**0.5)
        else:
            self.w1 = self.w1 - self.lr * self.dl_dw1
            self.w2 = self.w2 - self.lr * self.dl_dw2
            self.w3 = self.w3 - self.lr * self.dl_dw3

class MLP_ReLU():

    def __init__(self, units1=10, units2=10, units3=1):
        self.w1 = np.random.normal(0, 1, (2, units1))
        self.w2 = np.random.normal(0, 1, (units1, units2))
        self.w3 = np.random.normal(0, 1, (units2, units3))
        self.w1_sum = 0
        self.w2_sum = 0
        self.w3_sum = 0

    def forward(self, x):
        self.x = x
        self.a1 = x @ self.w1
        self.z1 = ReLU(self.a1)
        self.a2 = self.z1 @ self.w2
        self.z2 = ReLU(self.a2)
        self.a3 = self.z2 @ self.w3
        self.pred_y = sigmoid(self.a3)

        return self.pred_y

    def backward(self, y):
        '''
        backward propagation
        '''
        dl_dy = derivative_mse(y, self.pred_y)
        # print(dl_dy.shape)
        dy_da3 = derivative_sigmoid(self.pred_y)
        # print(dy_da3.shape)
        da3_dw3 = self.z2
        # print(da3_dw3.shape)
        G1 = dy_da3 * dl_dy
        self.dl_dw3 = da3_dw3.T @ G1

        da3_dz2 = self.w3
        dz2_da2 = derivative_ReLU(self.z2)
        da2_dw2 = self.z1
        G2 = dz2_da2 * (G1 @ self.w3.T)
        self.dl_dw2 = da2_dw2.T @ G2

        da2_dz1 = self.w2
        dz1_da1 = derivative_ReLU(self.z1)
        da1_dw1 = self.x
        G3 = dz1_da1 * (G2 @ self.w2.T)
        self.dl_dw1 = da1_dw1.T @ G3

    def update(self, lr, optimizer=''):
        '''
        Update weight
        '''
        self.lr = lr
        if optimizer == 'adagrad':
            self.w1_sum += (self.w1**2)
            self.w2_sum += (self.w2**2)
            self.w3_sum += (self.w3**2)
            self.w1 = self.w1 - self.lr * self.dl_dw1 / (self.w1_sum**0.5)
            self.w2 = self.w2 - self.lr * self.dl_dw2 / (self.w2_sum**0.5)
            self.w3 = self.w3 - self.lr * self.dl_dw3 / (self.w3_sum**0.5)
        else:
            self.w1 = self.w1 - self.lr * self.dl_dw1
            self.w2 = self.w2 - self.lr * self.dl_dw2
            self.w3 = self.w3 - self.lr * self.dl_dw3

class MLP_wo_activation():

    def __init__(self, units1=10, units2=10, units3=1):
        self.w1 = np.random.normal(0, 1, (2, units1))
        self.w2 = np.random.normal(0, 1, (units1, units2))
        self.w3 = np.random.normal(0, 1, (units2, units3))
        self.w1_sum = 0
        self.w2_sum = 0
        self.w3_sum = 0

    def forward(self, x):
        self.x = x
        self.a1 = x @ self.w1
        # self.z1 = ReLU(self.a1)
        self.a2 = self.a1 @ self.w2
        # self.z2 = ReLU(self.a2)
        self.pred_y = self.a2 @ self.w3
        # self.pred_y = sigmoid(self.a3)

        return self.pred_y

    def backward(self, y):
        '''
        backward propagation
        '''
        dl_dy = derivative_mse(y, self.pred_y)
        dy_dw3 = self.a2
        self.dl_dw3 = dy_dw3.T @ dl_dy

        dy_da2 = self.w3
        da2_dw2 = self.a1
        G2 = (dl_dy @ self.w3.T)
        self.dl_dw2 = da2_dw2.T @ G2

        da2_da1 = self.w2
        da1_dw1 = self.x
        G3 = (G2 @ self.w2.T)
        self.dl_dw1 = da1_dw1.T @ G3

    def update(self, lr, optimizer=''):
        '''
        Update weight
        '''
        self.lr = lr
        if optimizer == 'adagrad':
            self.w1_sum += (self.w1**2)
            self.w2_sum += (self.w2**2)
            self.w3_sum += (self.w3**2)
            self.w1 = self.w1 - self.lr * self.dl_dw1 / (self.w1_sum**0.5)
            self.w2 = self.w2 - self.lr * self.dl_dw2 / (self.w2_sum**0.5)
            self.w3 = self.w3 - self.lr * self.dl_dw3 / (self.w3_sum**0.5)
        else:
            self.w1 = self.w1 - self.lr * self.dl_dw1
            self.w2 = self.w2 - self.lr * self.dl_dw2
            self.w3 = self.w3 - self.lr * self.dl_dw3

def train(x, y, model, lr=1, epochs=10000, optimizer=''):
    loss_list = []
    epoch_list = []
    for epoch in range(epochs):
        pred_y = model.forward(x)
        loss = loss_mse(y, pred_y)
        dL = derivative_mse(y, pred_y)
        model.backward(y)
        model.update(lr, optimizer=optimizer)
        if (epoch+1) % 500 == 0:
            print(f'epoch {epoch+1} loss : {loss}')
        loss_list.append(loss)
        epoch_list.append(epoch+1)
    pred_y = model.forward(x)
    print(pred_y)
    show_result(x, y, pred_y)
    pred_y[pred_y > 0.5] = 1
    pred_y[pred_y <= 0.5] = 0
    print(f'Acc: {np.sum(pred_y == y) / y.shape[0] * 100}%')
    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

if __name__ == '__main__':

    # Linear data neural network with sigmoid functions
    x, y = generate_linear()
    model_linear = MLP(units1=10, units2=10)
    train(x, y, model_linear, lr=1)
    del model_linear

    # XOR data neural network with sigmoid functions
    x, y = generate_XOR_easy()
    model_XOR = MLP(units1=10, units2=10)
    train(x, y, model_XOR, lr=1)
    del model_XOR

    # Linear data neural network with ReLU functions
    x, y = generate_linear()
    model_linear = MLP_ReLU()
    train(x, y, model_linear)
    del model_linear

    # XOR data neural network with ReLU functions
    x, y = generate_XOR_easy()
    model_XOR = MLP_ReLU()
    train(x, y, model_XOR, lr=0.1, epochs=50000)
    del model_XOR

    # Linear data neural network without activation functions
    x, y = generate_linear()
    model_linear = MLP_wo_activation(units1=2, units2=2)
    train(x, y, model_linear, lr=0.001, epochs=10000)
    del model_linear

    # XOR data neural network without activation functions
    x, y = generate_XOR_easy()
    model_XOR = MLP_wo_activation(units1=4, units2=4)
    train(x, y, model_XOR, lr=0.01, epochs=10000)
    del model_XOR