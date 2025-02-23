import numpy as np
import matplotlib.pyplot as plt

def load_model(filepath='model/model_weights.npz'):
    npzfile = np.load(filepath)
    return npzfile['W1'], npzfile['b1'], npzfile['W2'], npzfile['b2']

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

data = np.loadtxt("data/test.csv", delimiter=",", skiprows=1)
Y_test = data[:, 0].astype(int)
X_test = data[:, 1:]

random_index = np.random.randint(0, X_test.shape[0])
X_example = X_test[random_index].reshape(-1, 1)
if X_example.shape[0] == 783:
    X_example = np.vstack((X_example, np.zeros((1, 1))))
true_label = Y_test[random_index]

W1, b1, W2, b2 = load_model()
_, _, _, A2 = forward_prop(W1, b1, W2, b2, X_example)
predictions = get_predictions(A2)
plt.gray()
plt.imshow(X_example.reshape(28, 28), interpolation='nearest')
plt.title(f"Prediction: {predictions[0]}")
# plt.show()