import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("BankNote_Authentication.csv")

X, y = df.iloc[:, :-1], df.iloc[:, -1]

X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)

# Fixed train_test_split call - parameters were in wrong order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def init_weights(n_x, n_h1, n_h2, n_y):
    w1 = np.random.randn(n_x, n_h1) * np.sqrt(2.0 / n_x)
    b1 = np.zeros((1, n_h1))
    w2 = np.random.randn(n_h1, n_h2) * np.sqrt(2.0 / n_h1)
    b2 = np.zeros((1, n_h2))
    w3 = np.random.randn(n_h2, n_y) * np.sqrt(2.0 / n_h2)
    b3 = np.zeros((1, n_y))

    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2,
                  "w3": w3,
                  "b3": b3}

    return parameters


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return Z > 0


def forward_propagation(X, parameters):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    w3 = parameters["w3"]
    b3 = parameters["b3"]

    # Fixed matrix multiplication
    Z1 = np.dot(X, w1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, w2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, w3) + b3
    A3 = 1 / (1 + np.exp(-Z3))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3
             }
    return A3, cache


def compute_cost(A3, Y):
    m = Y.shape[0]
    cost = -1 / m * np.sum(Y * np.log(A3 + 1e-8) + (1 - Y) * np.log(1 - A3 + 1e-8))
    cost = float(np.squeeze(cost))
    return cost


def backpropagation(X, Y, cache, parameters):
    m = X.shape[0]
    w1 = parameters["w1"]
    w2 = parameters["w2"]
    w3 = parameters["w3"]

    Z1 = cache["Z1"]
    A1 = cache["A1"]
    Z2 = cache["Z2"]
    A2 = cache["A2"]
    A3 = cache["A3"]

    dZ3 = A3 - Y
    dW3 = 1 / m * np.dot(A2.T, dZ3)
    db3 = 1 / m * np.sum(dZ3, axis=0, keepdims=True)

    dZ2 = np.dot(dZ3, w3.T) * relu_derivative(Z2)
    dW2 = 1 / m * np.dot(A1.T, dZ2)
    db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = np.dot(dZ2, w2.T) * relu_derivative(Z1)
    dW1 = 1 / m * np.dot(X.T, dZ1)
    db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2,
             "dW3": dW3,
             "db3": db3}

    return grads


def update_parameters(parameters, grads, learning_rate=0.01):
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    w3 = parameters["w3"]
    b3 = parameters["b3"]

    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    dW3 = grads["dW3"]
    db1 = grads["db1"]
    db2 = grads["db2"]
    db3 = grads["db3"]

    w1 = w1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    w3 = w3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2,
                  "w3": w3,
                  "b3": b3
                  }
    return parameters


def predict(parameters, X):
    A3, cache = forward_propagation(X, parameters)
    predictions = A3 > 0.5
    return predictions


def nn_model(X, Y, n_x, n_h1, n_h2, n_y, n_steps, learning_rate=0.01, print_cost=False):
    np.random.seed(42)
    parameters = init_weights(n_x, n_h1, n_h2, n_y)

    for i in range(n_steps):
        A3, cache = forward_propagation(X, parameters)
        cost = compute_cost(A3, Y)
        grads = backpropagation(X, Y, cache, parameters)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters


results = []

parameters_n_steps = [i for i in range(100, 1100, 100)]

for n_step in parameters_n_steps:
    parameters = nn_model(X_train, y_train, X_train.shape[1], n_h1=6,n_h2=6, n_y=1, n_steps=n_step, learning_rate=0.003,print_cost=False)
    predicts = predict(parameters, X_test)
    acc = accuracy_score(y_test.flatten(), predicts.flatten())
    results.append((n_step, acc))

# Sonuçları düzenli bir tablo olarak yazdırma
print("\n{:^10} {:^10}".format("n_step", "Accuracy"))
print("-" * 22)
for n_step, acc in results:
    print("{:^10} {:^10.4f}".format(n_step, acc))