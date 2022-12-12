import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report

dataIris = pd.read_csv(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, encoding='utf-8')

dataIris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

dataIris['Species'] = dataIris['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica':2})

X = dataIris.iloc[:100,[1,2,3]].values 
y = dataIris.iloc[:100,[4]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

class NeuralNetwork():

  def __init__(self):
    np.random.seed(1)
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.


        # We model a single neuron, with 4 input connections and 1 output connection.
        # We assign random weights to a 4 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
    self.synaptic_weights = 2*np.random.random((3,1)) - 1

    # The Sigmoid function, an S shaped curve,
    # normalizes the weighted sum of the inputs between 0 and 1.
  def sigmoid(self, x):
    return 1 /(1+np.exp(-x))

    # The derivative of the Sigmoid function.
    # The gradient of the Sigmoid curve
  def sigmoid_derivative(self,x):
    return x*(1-x)

    # The training phase adjusts the weights each time to reduce the error
  def train(self, training_inputs, training_outputs, training_iterations):

    for iteration in range(training_iterations):
      
      # Training inputs are processed
      output= self.think(training_inputs)

      # Calculate the error 
      error = training_outputs - output

      # Adjustments refers to the backpropagation process
      adjustments = np.dot(training_inputs.T, error*self.sigmoid_derivative(output))

      # Adjust the weights.
      self.synaptic_weights += adjustments
  
  # The neural network predicts new records.
  def think(self, inputs):
    # Pass inputs through our neural network (our single neuron).
    inputs = inputs.astype(float)
    output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

    return output

if __name__ == "__main__":
  
  #Assigning the perceptron to an object
  neural_network = NeuralNetwork()

#Printing synaptic weights before training
  print("Random synaptic weights:\n")
  print(neural_network.synaptic_weights)

  # The training set. We have 80 examples, each consisting of 4 input values
  # and 1 output value.
training_inputs = X_train
training_outputs = y_train

    # Train the neural network using a training set.
    # The number of iterations has been set to 1000
neural_network.train(training_inputs, training_outputs, 1000)

#Showing Synaptic weights after training
print("\nSynaptic weights after training:\n")
print(neural_network.synaptic_weights)

#Deploying Neuron on training data
predicted = neural_network.think(X_test)

#Transforming results into Pandas Dataframe
predicted_df = pd.DataFrame({'Result': predicted[:, 0]})

##Create a function to get a precise result from the Artificial Neuron

#If the score is higher than 0.5 then it's a 1 otherwise a 0
def getResult(score):
    if score < 0.5:
        return 0
    elif score >= 0.5:
        return 1

#Apply function on predicted dataframe
predicted_df = predicted_df.Result.apply(lambda x: getResult(x))

#Evaluate model performance
print("\n", classification_report(y_test, predicted_df))