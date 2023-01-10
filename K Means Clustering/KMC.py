import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics

"""
K Means Clustering 
    -- Unsupervised learning model
    -- Attempts to divide the data into section and predicts a given data point to a certain sections
    -- Centroids
        -- https://en.wikipedia.org/wiki/Centroid
    -- How to
        -- First Step: Group the data 
            -- Picks two random centroids
            -- Draws a straight line 90 degrees (perpendicular) between the two centroids 
            -- Then divides the points using a line drawn through the mid point and groups accordingly 
            -- Essentially, finds which data points are closest to a given centroid
        -- Second Step: Move the centroid
            -- Find the center of a given group of points
            -- Takes the average of all the given points in a group 
            -- Position the centroid at that location
        -- Third Step: Repeat steps one and two
            -- Re-assign the data points when repeating 
            -- Continue until there are no changes between the data points
                -- At this point the data points are clustered
    -- Some Disadvantages
        -- Speed
            -- Has to do it by the points * centroids * iterations * features
    -- Some Advantages
        --  Can work with multidatasets
"""

#Load data
digits = load_digits()

#Scales with in -1 to 1
data = scale(digits.data)

#Gets labels
y = digits.target

#Sets the amount of clusters
#k = len(np.unique(y)) is another option when working with multiple datasets
k = 10

#Get the amount of instances and features
samples, features = data.shape

"""
Scoring:
To score our model we are going to use a function from the sklearn website. It computes many different scores for different parts of our model.
"""
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)