import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data
import matplotlib.pyplot as plt

nnfs.init()

#The vertical data is just three groups of points each with their own class, this is basic data but easy to understand for the network
X, y = spiral_data(samples=100, classes=3)


class Layer_Dense:
    """
    This is a layer, basically output is inputs * weights + bias
    """
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    """
    Activates all the neurons in the middle of the network
    gets rid of negatives
    """
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_softmax:
    """
    Changes the outputs to be probabilities that all add to 1
    """
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

#how far from the expected value
class Loss:
    """
    The base class for loss functions
    Determines the average loss for a batch
    """
    def calculate(self,output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentrophy(Loss):
    """
    Calculates how well the predictions match the model
    """
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #basically (0, 1), the code/math just does not like 0
        
        #allows for multiple data types
        if len(y_true.shape) == 1: #if scaler values
            correct_confidences = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape) == 2: 
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

def AskToContinue():
    """
    Asks if the model seems good and we want to continue to be able to enter our own values
    """
    isContinue = input("\nModel Trained, would you like to continue? (Y/N) ")
    if isContinue == "Y" or isContinue == "y":
        ContinueToInput()
    if isContinue == "N" or isContinue == "n":
        print("Program closed")
    else:
        print("please enter Y or N")
        AskToContinue()
    
                  
def ContinueToInput():
    inputValues = input("Enter two coordinates (0-1), seperate them with a space: ")

    #make the data readible 
    user_data = np.array([float(i) for i in inputValues.strip().split()])
    user_data = user_data.reshape(1, 2) 

    #upload the trained model and pass the data through
    dense1.weights = best_dense1_weights
    dense1.biases = best_dense1_biases
    dense2.weights = best_dense2_weights
    dense2.biases = best_dense2_biases
    
    dense1.forward(user_data)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    #print the probabilities
    print(activation2.output[0])
    
#pass everythjng through the program
def ForwardPass():
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
def BackwardPass():
    activation2.backward(activation2.output, y)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

#Build and connect the actual layers of the network
dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_softmax()

loss_function = Loss_CategoricalCrossentrophy()

lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

#lists for plot
lossList = []
iList = []

#Training parameters
learning_rate = 0.1
iterations = 100000

#train the model using random weight mutations with the lowest loss
for iteration in range(iterations):
    ForwardPass()
    
    #Calculate loss
    loss = loss_function.calculate(activation2.output, y)
    
    BackwardPass()
    
    # Update weights and biases
    dense1.weights -= learning_rate * dense1.dweights
    dense1.biases -= learning_rate * dense1.dbiases
    dense2.weights -= learning_rate * dense2.dweights
    dense2.biases -= learning_rate * dense2.dbiases

    # Save best model
    if loss < lowest_loss:
        lowest_loss = loss
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
    
    if iteration % 1000 == 0:
        lossList.append(loss)
        iList.append(iteration)
        print(f"Iteration: {iteration} Loss: {loss}")
        
    
    
    

plt.plot(lossList,iList)


ForwardPass()

#print the top 5, each # is probability of that class ([prob of class 0   prob of class 1    prob of class 2])
print(activation2.output[:5])

#print out the chosen classes for each output
print(np.argmax(activation2.output[:5], axis=1))

#give me the loss amount, closer to 0 means more accurate
loss = loss_function.calculate(activation2.output, y)
print("Loss: ", loss)

plt.show()
AskToContinue()

    