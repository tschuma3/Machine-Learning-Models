import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

"""
Neural Networks
    -- How it works
        -- It is like a brain
        -- Has neurons and connections
            -- Connected neurons tells other connected neurons to fire or not fire
        -- It is a layer system
        -- Fully Connected Neural Network
            -- Each neuron from one layer is connected to each neuron in the next layer exactly once
        -- Weights
            -- Each connection between neurons has a weight
        -- Only really care about the output
        -- Can have 4 inputs and 25 outputs or 4 inputs and 1 output
        -- Snake Example
            -- Goal
                -- For the snake to survive
            -- First step is to decide the input(s) and output(s)
                -- Inputs
                    -- Is there something in each cardinal direction
                    -- Recommended direction
                -- Output
                    -- Should the snake follow the recommended direction
        -- Designing and Architecture
            -- Each input neuron has a value
            -- Each connection has a weight
            -- Determining the output is taking the weighted sum of the values * the weights + the bias
                -- Dot Product
                    -- https://en.wikipedia.org/wiki/Dot_product
                    -- E(Vi * Wi) + Bi
                        -- E == summation
                        -- Vi == value i
                        -- Wi == weight i
                        -- Bi == bias
                -- The bias is some constant value for each of the weights 
        -- Training
            -- Remeber the weights and biases
            -- Then adjust the weights and the biases after each iteration
            -- Do this till the it gives a high accuracy
        -- Activation Functions
            -- Nonlinear function that allows you to add a degree of complexity to your network so the graph representation will look more like sin, cos, or tan instead of a straight line
            -- Not linear therefor adding a higher degree of complexity
            -- Example Functions
                -- Sigmoid
                    -- https://en.wikipedia.org/wiki/Sigmoid_function
                    -- Map any value give to be between -1 and 1
                    -- The closer the value is to infinity the closer it is to 1 and the closer the value is to negative infinity the closer it is to -1
                -- Rectify Linear Unit (Rectifier)
                    -- https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
                    -- Takes all the values that are negative and makes them 0
                    -- Takes all the positive values and makes them more positive
                    -- Essentially puts all the values between 0 and 1
            -- Shrinks down the data so that it is not as large
                -- Normalizes the data
            -- Applying Activation Function Example
                -- f(x)[E(Vi * Wi) + Bi]
                    -- f(x) == the activation function
                        -- f(x) will now be the output
        -- Loss Function
            -- Helps adjusting the weights and biases
            -- Calculates error
            -- Many different loss functions
            -- Not linear therefor adding a higher degree of complexity
        -- Hidden Layer(s)
            -- Can have more of everything (weights, biases, neurons, etc) to have more accurate networks
        -- Don't want to pass all the data at once
            -- This will alleviate the program from memorizing verses analyzing
        -- Flattening the data
            -- Taking any interior list and squishing them together
                -- [[1], [2], [3]] ----> [1, 2, 3]
        -- Hypertraining
            -- Changing weights, biases, etc in a small way to get a more accurate results
"""
#region Loading and Looking at Data

#Import data
data = keras.datasets.fashion_mnist

#Sets the data into respected arrays using keras
#If not using a library like keras, then would have to use and write own arrays, for loops, etc to bring the data in
(train_images, train_labels), (test_images, test_labels) = data.load_data()

#Creates a list of labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Reduce the size of the data
train_images = train_images / 255.0
test_images = test_images / 255.0

#Showing an image
"""
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()
"""

#endregion

#region Creating a Model

#Define the architectire or a sequence of layers
    #Flatten == Normalize the data so it is passable
    #Dense == A fully connected layer
        #Softmax == Essentially the probibility of the network thinking the input is something
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

#Setting up parameters for the model
    #Adam is a standard optimizer
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Training the model
    #Epochs == How many times the model will see an image
        #Normally need to tweek
model.fit(train_images, train_labels, epochs=5)

#Don't need because of the prediction function
"""
#Seeing how the model does
#Checking how the model does on the testing images and labels
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Tested Accuracy {test_accuracy}")
"""

#endregion

#region Making Predictions

#Creates an list of the predictions made
    #Shows for each of the thoughts for each clothing piece 
prediction = model.predict(test_images)

#Loops through the images and shows some images and there respected prediction
for i in range(5):
    #Shows actual value, predicted value, and image
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediciton: " + class_names[np.argmax(prediction[i])]) #np.argmax gives the largets value
    plt.show()

#endregion