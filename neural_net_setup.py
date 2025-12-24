import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm #shows progress bar   

from sklearn.datasets import fetch_openml
#mnist is a dataset of 70 000 handwritten digits (0-9)

mnist = fetch_openml(name='mnist_784', version=1)

print(mnist.keys())

data = mnist["data"]
labels = mnist["target"]


#picking random datapoint to know image and its shape
n = np.random.choice(np.arange(data.shape[0]+1))

print(n)
#n = 252247 

test_img = data.iloc[n].values
test_lable = mnist.target.iloc[n]

print(test_img.shape) #(784,)

side_length = int(np.sqrt(test_img.shape))
reshaped_test_img = test_img.reshape(side_length, side_length)


print("Image label:", test_lable) #9

plt.imshow(reshaped_test_img, cmap='gray')
plt.axis('off')
plt.show()
#the shape will be (28, 28) because sqrt(784) = 28, so the image is 28 pixels by 28 pixels/matrix

#now NN with single deep layyer and 4 neurons

w1 = np.ones((784, 4)) *0.01 #weights for layer 1
z1 = np.dot(data,w1) #pre-activation output of layer 1
print("z1 shape:", z1.shape) #(70000, 4)

w2 = np.ones((4, 10)) #weights for layer 2
z2 = np.dot(z1, w2) #pre-activation output of layer
print("z2 shape:", z2.shape) #(70000, 10)

#activation functions

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def leaky_relu(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, z, z * 0.01)


def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z)) #subtracting max for numerical stability
    return exp_z / np.sum(exp_z, axis=0)


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))


#now need derivative of the activation function to perform the gradient descent
def derivative(function_name:str, z:np.ndarray) -> np.ndarray:
    if function_name == 'sigmoid':
        sig = sigmoid(z)
        return sig * (1 - sig)
    elif function_name == 'relu':
        y = (z > 0) * 1
        return y
    elif function_name == 'tanh':
        return 1 - np.square(tanh(z))
    elif function_name == 'leaky_relu':
        return np.where(z > 0, 1, 0.01)
    else:
        return "No such activation function exists"
    



#making the neural network class

class NeuralNetwork:
    def __init__(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, activation: str, num_labels: int, architecture: List[int]):
        self.X = normalize(X) # normalize training data in range 0,1
        assert np.all((self.X >= 0) | (self.X <= 1)) # test that normalize succeded
        self.X, self.X_test = X.copy(), X_test.copy()
        self.y, self.y_test = y.copy(), y_test.copy()
        self.layers = {} # define dict to store results of activation
        self.architecture = architecture # size of hidden layers as array  
        self.activation = activation # activation function
        assert self.activation in ["relu", "tanh", "sigmoid", "leaky_relu"]
        self.parameters = {}
        self.num_labels = num_labels
        self.m = X.shape[1]
        self.architecture.append(self.num_labels)
        self.num_input_features = X.shape[0]
        self.architecture.insert(0, self.num_input_features)
        self.L = len(architecture) 
        assert self.X.shape == (self.num_input_features, self.m)
        assert self.y.shape == (self.num_labels, self.m)


    def initialize_parameters(self):
        for i in range(1, self.L):
            print(f"Initializing weights and biases for layer {i}")
            self.parameters['w' + str(i)] = np.random.randn(self.architecture[i], self.architecture[i-1]) * 0.01 #initialize weights small random values
            self.parameters['b' + str(i)] = np.zeros((self.architecture[i], 1)) #initialize biases to zero


#feedforward - used to compute the output of the neural network given the input data
    
    def forward(self):
            params=self.parameters
            self.layers["a0"] = self.X
            for l in range(1, self.L-1):
                self.layers["z" + str(l)] = np.dot(params["w" + str(l)], 
                                                self.layers["a"+str(l-1)]) + params["b"+str(l)]
                self.layers["a" + str(l)] = eval(self.activation)(self.layers["z"+str(l)])
                assert self.layers["a"+str(l)].shape == (self.architecture[l], self.m)
            self.layers["z" + str(self.L-1)] = np.dot(params["w" + str(self.L-1)],
                                                    self.layers["a"+str(self.L-2)]) + params["b"+str(self.L-1)]
            self.layers["a"+str(self.L-1)] = softmax(self.layers["z"+str(self.L-1)])
            self.output = self.layers["a"+str(self.L-1)]
            assert self.output.shape == (self.num_labels, self.m)
            assert all([s for s in np.sum(self.output, axis=1)])        
            
            cost = - np.sum(self.y * np.log(self.output + 0.000000001))

            return cost, self.layers


#backpropagation - used to compute gradients by backpropagating the error
    
    def backpropagate(self):
            derivatives = {}
            dZ = self.output - self.y
            assert dZ.shape == (self.num_labels, self.m)
            dW = np.dot(dZ, self.layers["a" + str(self.L-2)].T) / self.m
            db = np.sum(dZ, axis=1, keepdims=True) / self.m
            dAPrev = np.dot(self.parameters["w" + str(self.L-1)].T, dZ)
            derivatives["dW" + str(self.L-1)] = dW
            derivatives["db" + str(self.L-1)] = db
            
            for l in range(self.L-2, 0, -1):
                dZ = dAPrev * derivative(self.activation, self.layers["z" + str(l)])
                dW = 1. / self.m * np.dot(dZ, self.layers["a" + str(l-1)].T)
                db = 1. / self.m * np.sum(dZ, axis=1, keepdims=True)
                if l > 1:
                    dAPrev = np.dot(self.parameters["w" + str(l)].T, (dZ))
                derivatives["dW" + str(l)] = dW
                derivatives["db" + str(l)] = db
            self.derivatives = derivatives
            
            return self.derivatives
    
#fit  - used to train the neural network
    def fit(self, lr=0.01, epochs=1000):
        self.costs = [] 
        self.initialize_parameters()
        self.accuracies = {"train": [], "test": []}
        for epoch in tqdm(range(epochs), colour="BLUE"):
            cost, cache = self.forward()
            self.costs.append(cost)
            derivatives = self.backpropagate()            
            for layer in range(1, self.L):
                self.parameters["w"+str(layer)] = self.parameters["w"+str(layer)] - lr * derivatives["dW" + str(layer)]
                self.parameters["b"+str(layer)] = self.parameters["b"+str(layer)] - lr * derivatives["db" + str(layer)]            
            train_accuracy = self.accuracy(self.X, self.y)
            test_accuracy = self.accuracy(self.X_test, self.y_test)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch:3d} | Cost: {cost:.3f} | Accuracy: {train_accuracy:.3f}")
            self.accuracies["train"].append(train_accuracy)
            self.accuracies["test"].append(test_accuracy)
        print("Training terminated")



#predict - used to predict labels for new data points after training


    def predict(self, x):
            params = self.parameters
            n_layers = self.L - 1
            values = [x]
            for l in range(1, n_layers):
                z = np.dot(params["w" + str(l)], values[l-1]) + params["b" + str(l)]
                a = eval(self.activation)(z)
                values.append(a)
            z = np.dot(params["w"+str(n_layers)], values[n_layers-1]) + params["b"+str(n_layers)]
            a = softmax(z)
            if x.shape[1]>1:
                ans = np.argmax(a, axis=0)
            else:
                ans = np.argmax(a)
            return ans
    

#accuracy - used to compute accuracy of the model on given data 
    def accuracy(self, X, y):
        P = self.predict(X)
        return sum(np.equal(P, np.argmax(y, axis=0))) / y.shape[1]*100
    




#training and testing split

train_test_split_no = 60000
X_train = data.values[:train_test_split_no].T
y_train = labels[:train_test_split_no].values.astype(int)
y_train = one_hot_encode(y_train, 10).T
X_test = data.values[train_test_split_no:].T
y_test = labels[train_test_split_no:].values.astype(int)
y_test = one_hot_encode(y_test, 10).T
X_train.shape, X_test.shape
((784, 60000), (784, 10000))


#initializing and training the neural network

PARAMS = [X_train, y_train, X_test, y_test, "relu", 10, [128, 32]]
nn_relu = NeuralNetwork(*PARAMS)
epochs_relu = 200
lr_relu = 0.003
nn_relu.fit(X_train, y_train, lr=lr_relu, epochs=epochs_relu)
nn_relu.plot_cost(lr_relu)