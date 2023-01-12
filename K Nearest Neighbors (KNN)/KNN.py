import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import numpy as np

"""
K Nearest Neighbors
    -- Used for classifications, typically
        -- 4 Classes Example
            1. unacc
            2. acc
            3. good
            4. vgood
    -- Need to convert non numerical data into numerical data
    -- Attempts to classify the data based on the given data
    -- Looks for grouping of data points and groups that point to the closest group
    -- Finds k neighbors and basically majority rules in where the data point is grouped
    -- Picks an odd value for k so there are no ties
    -- The math
        -- Draws a line from the data point to the closest neighbors
        -- Finds the magnitude of the line
            -- The one of the defult and one of the simplest ways is Euclidean Distance
            -- Euclidean Distance: https://en.wikipedia.org/wiki/Euclidean_distance
    -- Has to save every data point because it needs to look at all of them for each iteration
    -- Works in linear time
"""

#Gets the data file
data = pd.read_csv(r"D:\GitHub Repos\Machine-Learning-Models\K Nearest Neighbors (KNN)\car.data")

#Takes the labels and encods them into numerical values
le = preprocessing.LabelEncoder()

#Gets the respected column, turn them into a list, and transform them into numerical values
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clss = le.fit_transform(list(data["class"]))

predict = "class"

#X is attributes or features and y is labels
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(clss)

#Taking all the attributes and lables, splitting them up in 4 arrays
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

#Create classifier with the amount of neighbors wanted
model = KNeighborsClassifier(n_neighbors=9)

#Trains and prints accuracy
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy}")

#Predicts
predicted = model.predict(x_test)

#Prints the data points, prediction, and actual value
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print(f"Predicted: {names[predicted[x]]}, Data: {x_test[x]}, Actual: {names[y_test[x]]}")

    #Index and look at the points
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)