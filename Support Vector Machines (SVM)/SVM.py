import sklearn
from sklearn import datasets 
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClas #Implement KNN if wanting to see the difference between KNN and SVM
import pandas as pd
import numpy as np

"""
Support Vector Machines
    -- Attempts to create a hyperplane
    -- Margins
        -- They are support vectors
    -- Hyperplane 
        -- It is a linear way to divide data
        -- Find where the hyperplane is the same distance from the closest points of each of the opposing classes
        -- Can generate an infinite amount of hyperplanes
        -- How to pick the best hyperplane
            -- The the two closest point from each class is the furthest possible distance (d) from the line
                -- Can't find a distance greater
            -- Why do we want the distance to be the largest
                -- It will be classified more accurately
                -- The largert the distance and the larger the margin the more accurate it will be
    -- Problems 
        -- Scattered data and trying to implement a hyperplane based off that data
        -- Solution
            -- Kernal
                -- Basically a function
                    -- Different kinds of kernals
                    -- Example: x1^2 + x2^2 --> x3
                -- Can transform the data from 2D to 3D
                -- function(x1, x2) --> x3
                -- Hyperplanes work the same in 3D as it does in 2D
                -- If the hyperplane is still difficult to find, repeat the conversion from 2D to 3D, but instead move up a dimension 
    -- Soft Margin 
        -- Allows for outlier points to exist within the margins
    -- Hard Margin
        -- Can't have any outlier points to exist within the margins
"""

#Imports the dataset
cancer = datasets.load_breast_cancer()

#Sets up the training variables
x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']

#Implement classifier
#SVC takes many parameters to better predict the data
#kernel, C == soft margin
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

#Y prediction
y_pred = clf.predict(x_test)

#Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)

print(accuracy)