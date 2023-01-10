import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

"""
Linear Regression
    -- Looks at a scatter of data points and tries to find a best fit line
    -- When data directly correlates to each other
    -- Line can be defined as y = mx + b
    -- The line with multidata will be in multidimensional space
"""


#The csv file we want to get the data from
data = pd.read_csv(r"D:\GitHub Repos\Machine-Learning-Models\student-mat.csv", sep=";")

#What data we want to use
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#What we want to predict
predict = "G3"

#All features or attributes
X = np.array(data.drop([predict], 1))

#All of the labels of the dataset
y = np.array(data[predict])

#Taking all the attributes and lables, splitting them up in 4 arrays
#Can't train off or previous data, as it already know the answers
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

#Uncomment for first iteration
"""
#Sets a best model and loops a set amount of times
best = 0
for _ in range(30):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    #Training model
    linear = linear_model.LinearRegression()

    #Fit the data to find a best fit line storing in linear
    linear.fit(x_train, y_train)

    #Finds the accuracy and prints to the console
    accuracy = linear.score(x_test, y_test)
    print(f"Accuracy: {accuracy}")

    #Only takes the model with the best accuracy
    if accuracy > best:
        best = accuracy

        #Saving a pickle file to be used
        with open("student_model.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

#Reading in pickle file
pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)

#Prints out the coefficient and y intercept
print(f"Coefficient: {linear.coef_}")
print(f"Intercept: {linear.intercept_}")

#Predicting on a real student
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#Plotting on a grid using a scatter plot
p = "G1"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()