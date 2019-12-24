import random

import numpy as np
import pandas as pd

from Cluster import KNN
from Data import CATEGORICAL_DICTIONARY, DataConverter


class Kmeans(KNN):

    def __init__(self, k_val, data_instance):
        super().__init__(k_val, data_instance)
        self.converter = DataConverter()

    def k_means(self, data_set, k_val):  # Method for K-Means
        print("\n----------------- Running K-Means -----------------")

        data_space = self.get_data_space(data_set)
        clusters = self.set_clusters_to_dict(data_set, k_val, data_space)
        return self.cluster_data(clusters, data_set)

    def k_random_point(self, data_set, k_val, data_space):  # Method to grab k_random rows for centroid method
        data_set = data_set
        centroid_points = []  # List of centroid points type Series
        # Following row iteration with iteritems() sourced from https://stackoverflow.com/questions/28218698/how-to-iterate-over-columns-of-pandas-dataframe-to-run-regression/32558621 User: mdh and mmBs
        for k in range(k_val):  # Grabs k Centroids
            current_point = []  # List for current random point in loop
            for item in range(data_set.shape[1]):  # Loop through each item in the row
                # current_point.append(random.uniform(0, float(len(CATEGORICAL_DICTIONARY)))) # radom uniform source: https://pynative.com/python-get-random-float-numbers/
                current_point.append(random.uniform(data_space[item][0], data_space[item][
                    1]))  # radom uniform source: https://pynative.com/python-get-random-float-numbers/

            centroid_points.append(current_point)  # Appends the point to a list to be returned

        return centroid_points  # Returns a list of centroid points

    def predict_centroids(self, centroids, data_set):  # Method to return closest cluster to test data

        for _, data in data_set[data_set].iterrows():  # Loops through the rows of the data set
            distance = None  # Initializes distance
            closest_centroid = None  # Keeps track of the current closes centroid cluster
            closest_centroid_euclidian_distance = None  # Keeps track of the closest euclidian distance.
            cluster_val = 1
            for centroid in centroids:  # Loops through the k centroid points
                euclid_distance = KNN.get_euclidean_distance(centroid,
                                                             data)  # Gets the distance between the centroid and the data point

                if distance is None or euclid_distance < distance:  # Updates the distance to keep track of the closest point
                    distance = euclid_distance
                    # closest_centroid = centroid
                    closest_centroid = cluster_val
                    closest_centroid_euclidian_distance = distance
                cluster_val += 1
            # Print closest cluster to the test data point.
            # print("\nEuclidian Distance to Closest K-Means Cluster: ", closest_centroid_euclidian_distance)
            # print("Closest Cluster: Cluster ", closest_centroid)

    def set_clusters_to_dict(self, data_set, k_val, data_space):  # Create an initial dictionary of cluster points
        while (True):
            clusters = {}
            duplicate_points = False
            points = self.k_random_point(data_set, k_val, data_space)  # Get random k-means points
            keys = 0
            for cluster in points:  # Loop through the random cluster points and append them to the dictionary
                if cluster in clusters.values():  # Checks if random values created the same cluster
                    duplicate_points = True
                else:
                    clusters[keys] = cluster  # Creates the dictionary of cluster points
                keys += 1
            if duplicate_points is False:
                return clusters

    def cluster_data(self, clusters, data_set):  # Loop until clusters have converged
        previous_clusters = []  # Initializes to check if previous value mached
        while (True):
            current_clusters = []
            for point in range(len(clusters)):  # Appends an empty list
                current_clusters.append([])

            for _, value in data_set.iterrows(): # Loop rows of the data set
                cluster_key = 0 # Appends a key for the closest value of the dictionary
                closest_point = [None, float('inf')]  # Index of dictionary, distance value
                value = list(value) # Won't work without this
                for row in clusters.values(): # Loops through the values in the cluster to compare distance
                    distance = KNN.get_euclidean_distance(row, value) # Gets the euclidean distance
                    if distance < closest_point[1]: # Checks if it is closer than the previous closest point
                        closest_point = [cluster_key, distance] # Sets the closest point
                    cluster_key += 1
                current_clusters[closest_point[0]].append(value) # Appends the closest point to a the corresponding cluster

            clusters = self.mean_clusters(current_clusters, data_set) # Gets the updated k-mean clusters
            if previous_clusters == current_clusters:
                print('-------------------------- K-Means has converged ------------------')
                cluster_list = []
                for cluster in clusters.values():# Convert the k-means points to a list
                    cluster_list.append(cluster)
                return cluster_list
            previous_clusters = current_clusters
    def mean_clusters(self, updated_clusters, data_set): # Get the new mean clusters
        updated_centroids = {}
        for centroid_key in range(len(updated_clusters)): # creates an empty dictionary to return back
            updated_centroids[centroid_key] = []
        dict_key = 0
        new_cluster_point = [0] * (data_set.shape[1]) # Creates a new empty dictionary for a point to add to the cluster
        for data in updated_clusters:
            columns = (len(data)) # Length of the columns.
            if columns == 0: # Handles a division by zero case
                pass
            else:
                for data_point in data: # Loops through the cluster data
                    cluster_iterator = 0
                    for point in data_point:
                        new_cluster_point[cluster_iterator] += point # Sums the columns
                        cluster_iterator += 1

                for point in range(len(new_cluster_point)):
                    new_cluster_point[point] /= columns # Devides for the mean
            updated_centroids[dict_key] = new_cluster_point # Append the mean points to a dictionary
            new_cluster_point = [0] * (data_set.shape[1]) # Clear new cluster list for next cluster
            dict_key += 1
        return updated_centroids

    def get_data_space(self, data_set): # Gets the dataspace min/max for the columns
        data_space = []
        for col in range(data_set.shape[1]):
            data_space.append([data_set[col].min(), data_set[col].max()]) #
        return data_space
