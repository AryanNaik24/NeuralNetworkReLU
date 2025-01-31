import numpy as np
# Code written by @AryanNaik24 on github
# link to github github.com/AryanNaik24


# Building Neural Network From Scratch



# Creating all activation functions first

# relu is Rectified Linear unit that introduce non linearity
def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x>0).astype(float)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)


# Initialize all the weights an biases
def initilize_weights_biases(layers):
    np.random.seed(42)
    weights={}

    for i in range(1, len(layers)):
        weights[f"W{i}"] = np.random.randn(layers[i-1], layers[i]) * np.sqrt(2 / layers[i-1])  # He Initialization
        weights[f"b{i}"] = np.zeros((1, layers[i]))


    return weights

# Forward propogation
def forward_propagation(X,weights):
    activation = {"A0":X}
    L = len(weights)//2

    for i in range(1,L):
        Z = np.dot(activation[f"A{i-1}"],weights[f"W{i}"])+weights[f"b{i}"]
        A=relu(Z)

        activation[f"Z{i}"],activation[f"A{i}"]=Z,A


    # Output Layer (uses sigmoid activation)
    Z_final = np.dot(activation[f"A{L-1}"],weights[f"W{L}"])+weights[f"b{L}"]
    A_final = sigmoid(Z_final)
    activation[f"Z{L}"],activation[f"A{L}"]=Z_final,A_final

    return activation

# Backward propagation

def backward_propagation(X,Y,weights,activation):
    gradients = {}
    L = len(weights)//2
    m= X.shape[0]

    # output layer gradient
    dZ = activation[f"A{L}"]-Y
    for i in range(L, 0, -1):
        dW = np.dot(activation[f"A{i-1}"].T,dZ)/m
        db = np.sum(dZ,axis=0,keepdims=True)/m
        gradients[f"dW{i}"], gradients[f"db{i}"] = dW, db

        if i>1:
            dZ = np.dot(dZ, weights[f"W{i}"].T) * relu_derivative(activation[f"Z{i-1}"])

    return gradients

# Update weights
def update_weights(weights, gradients, learning_rate):
    for key in weights.keys():
        weights[key] -= learning_rate * gradients["d" + key]
    return weights



# Train on data
def train (X,Y ,layers, learning_rate=0.1, epochs=500):
    weights = initilize_weights_biases(layers)

    for i in range(epochs):
        activation = forward_propagation(X,weights)
        gradient = backward_propagation(X,Y,weights,activation)
        weights = update_weights(weights,gradient,learning_rate)

        if i % 500 == 0:
            loss = -np.mean(Y * np.log(activation[f"A{len(layers)-1}"]) + (1 - Y) * np.log(1 - activation[f"A{len(layers)-1}"]))
            print(f"Epoch {i}: Loss = {loss:.4f}")
    return weights


# Make predictions
def predict(X, weights):
    activation = forward_propagation(X, weights)
    return (activation[f"A{len(weights) // 2}"] > 0.5).astype(int)


# Random XOR problem from internet
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

layers = [2, 10, 8, 5, 3, 1]  # 2 is input layers 1 is output layer all in between are hiddenlayers
weights = train(X, Y, layers, learning_rate=0.1, epochs=5000)

predictions = predict(X, weights)
print("Predictions:", predictions.flatten())


# i will make one with tanh as well for small scale
