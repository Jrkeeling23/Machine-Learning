import random

import pandas as pd
import numpy as np
from process_data import Data


class KNN:
    """
    Anything to do with k-nearest neighbor should be in here.
    """

    def __init__(self):
        self.current_data_set = None
        self.data = None

    def perform_knn(self, query_point, train_data, k_val, name, in_data):
        """
        Function performs KNN to classify predicted class.
        :param in_data: the data instance from main to call process data functions
        :param name:  name of the data_set
        :param k_val: number of neighbors
        :param query_point: all data to compare an example from test_data too.
        :param train_data:  all data to "query" and predict
        :return: Predicted class
        """
        self.current_data_set = name
        self.data = in_data
        print("\n-----------------Performing KNN-----------------")
        distance_dict = {}  # place all indexes, which are unique, and distances in dictionary
        distance_list = []  # holds the k-number of distances
        label_list = []  # holds the k-number of labels associated with disances
        for index, row in train_data.iterrows():  # iterate through all data and get distances
            distance = (self.euclidean_distance(query_point, row))  # all features of x to a euclidean.
            distance_dict[index] = distance

        count = 0  # stops for loop
        for key, value in sorted(distance_dict.items(), key=lambda item: item[1]):
            # key is the index and value is the distance. Ordered least to greatest by sort().
            # if statement to grab the k number of distances and labels
            if count > k_val:
                break
            elif count is 0:
                count += 1  # first value is always 0.
                continue
            else:
                distance_list.append(value)  # add distance
                label_list.append(train_data.loc[key, self.data.get_label_col(self.current_data_set)])  # add label
                count += 1
        # TODO: get rid of prints, only needed to show you all the structure.
        print("Distance List: ", distance_list)
        print('Label list', label_list)

        print(str(k_val), "Nearest Neighbors (Class) to Query Point: ", label_list)

        return self.predict_by_distance(label_list)  # return the predicted values

    def euclidean_distance(self, query_point, comparison_point):
        """
        With multi dimensions: sqrt((x2-x1)+(y2-y1)+(z2-z1)+...))
        :param query_point: Testing example.
        :param comparison_point: example in training data.
        :return: float distance
        """

        # print("\n-----------------Getting Euclidean Distances-----------------")
        temp_add = 0  # (x2-x1)^2 + (y2 - y1)^2 ; addition part
        for feature_col in range(len(query_point)):

            if self.data.get_label_col(self.current_data_set) is feature_col:
                continue
            if type(query_point[feature_col]) is float or type(query_point[feature_col]) is int or type(
                    query_point[feature_col]) is np.float64:
                temp_sub = (query_point[feature_col] - comparison_point[feature_col]) ** 2  # x2 -x1 and square
                temp_add += temp_sub  # continuously add until square root

        return temp_add ** (1 / 2)  # square root

    def predict_by_distance(self, label_list):
        """
        Determines the prediction of class by closest neighbors.
        :param label_list: k-number of labels associated with distance list
        :param distance_list: k-number of closest distances
        :return: Predicted class
        """
        print("\n-----------------Deciding Predicted Nearest Neighbor-----------------")
        loop_iterator_location = len(label_list)  # Variable changes if nearest neighbor conflict.
        while True:
            nearest_neighbor = label_list[0]  # Sets the current pick to the first value in the list
            predict_dictionary = {}  # Temp dictionary to keep track of counts
            for class_obj in label_list[
                             :loop_iterator_location]:  # Loops through the input list of labels to create a dictionary with values being count of classes
                if class_obj in predict_dictionary.keys():  # Increases count if key exists
                    predict_dictionary[class_obj] += 1
                    if predict_dictionary[nearest_neighbor] < predict_dictionary[class_obj]:
                        nearest_neighbor = class_obj  # Sets the nearest neighbor to the class that occurs most.
                else:
                    predict_dictionary[class_obj] = 1  # Create key and set count to 1
            check_duplicates = list(predict_dictionary.values())  # Create a list to use the count function
            if check_duplicates.count(predict_dictionary[
                                          nearest_neighbor]) == 1:  # Breaks out of loop if the count of the top class occurrences is the only class sharing that count
                break
            else:
                loop_iterator_location -= 1  # By reducing the loop iterator, we remove the furthest neighbor from our counts.
        print("Predicted Nearest Neighbor: ", nearest_neighbor)
        return nearest_neighbor

    def edit_data(self):
        """
        Edit values for edit_knn by classifying x_initial; if wrong, remove x_initial. (option1)
        OR... if correct remove (option 2)
        :return: Edited values back to KNN
        """
        # TODO: edit data according to pseudo code from class on 9/23
        pass

    def condense_data(self, data_set, k_val, name, in_data):
        """
        Condense the data set by instantiating a Z = None. Add x_initial to Z if initial class(x_initial) != class(x)
        where x is an example in Z.
        So: Eliminates redundant data.
         :param dataSet: data we want to reduce.
         :param k_val: # of neighbors, used when performing knn
        :return: condensed data
        """
        # TODO: edit data according to pseudo code from class on 9/23
        print("\n-----------------Performing Condensed Dataset Reduction-----------------")
        self.data = in_data
        self.current_data_set = name
        # new dataset to hold condensed values
        # condensed_data = pd.DataFrame()
        firstElem = []  # use later to store values to remake dataset
        list_for_adding = []
        list_for_adding.append(firstElem)
        for val in data_set.iloc[0]:
            firstElem.append(val)
        col_list = list(data_set.columns)

        # finally got adding 1 row down
        condensed_data = pd.DataFrame([firstElem], columns=col_list)
        # condensed_data = condensed_data.append(firstElem)

        has_changed = True  # bool to break if condensedData no longer changes
        condensed_size = len(condensed_data.index)  # var to keep track of size of condensed data
        # add first found example to the data set (assuming [0][:] is valid here

        while has_changed is True:  # outside loop for CNN

            lowest_distance = 99999999  # holding distance here, settting to 999 just to make sure we get a smaller num
            minimum_index = -1  # index for that minimum element
            # go through every point in the data set, get point with lowest distance with class != to our example
            # TODO:  May want to make this random, not sure if needed
            for index, row in data_set.iterrows():
                # go through the condensed dataset and find point with the lowest distance to point from actual data (euclidian)
                for c_index, c_row in condensed_data.iterrows():  # should start with at least one
                    #  print(c_row)

                    e_dist = self.euclidean_distance(row, c_row)  # take distance
                    if e_dist < lowest_distance:  # compare current dist to last seen lowest
                        lowest_distance = e_dist  # store lowest distance
                        minimum_index = c_index  # store minimum index to check classification
                        # classify our 2 vals with KNN and compare class values
                # selecting value found an minimum index for condensed, and using the row we are iterating on
                condensed_value = self.perform_knn(condensed_data.iloc[minimum_index][:], data_set, k_val,
                                                   self.current_data_set, self.data)
                data_set_value = self.perform_knn(row, data_set, k_val, self.current_data_set, self.data)
                # compare the classes of the two predicted values
                # this assumes we get examples back that we need to select class from KNN
                #  TODO:  change this as needed by KNN algo
                if condensed_value != data_set_value:
                    # create new data set with new values
                    print("\n-----------------Adding datapoint to condensed dataset-----------------")
                    # add new values to list and append that 2 list for condensed data
                    vals = []
                    for val in row:
                        vals.append(val)
                    list_for_adding.append(vals)

                    condensed_data = pd.DataFrame(list_for_adding)

                    # print(len(condensed_data.index))
                    # print(condensed_size)

            # checking if the size of the condense dataset has changed, if so keep going, if not end loop
            if condensed_size is len(condensed_data.index) or len(condensed_data.index) > 100:
                has_changed = False  # if the length Has not changed, end loop
                break
            elif condensed_size > 10000:  # just in case break condition TODO: possibly remove
                print("in elif")
                has_changed = False
                break
            else:
                has_changed = True  # size has changed, keep going
                condensed_size = len(condensed_data.index)  # update our length

            # another break
        # print(has_changed)
        # if has_changed is False:
        #    print("in final break 2")
        #    break

        print("\n-----------------Finished performing Condensed Dataset Reduction-----------------")
        return condensed_data

    def centroids(self, data_set, k_val):  # Method for K-Means
        data_set = data_set[self.current_data_set]
        print("\n-----------------Starting K-Means Function-----------------")
        # centroid_points = self.create_initial_clusters(self.k_random_rows(data_set,
        #                                                                   k_val))  # Get random rows for centroid points then create the initial centroid point pd.DataFrames
        centroid_points = self.k_random_point(data_set, k_val)

        while True:
            current_points = []  # A list of the current data points for a cluster
            previous_points = centroid_points  # Sets a previous value to check if K-means has converged
            clusters = []  # List of clusters for use below
            initiate_list = 0
            for point in centroid_points:  # Make a list of DataFrame clusters
                clusters.append([])  # Instantiates a list for the clusters
                clusters[initiate_list].append(point)  # Adds the centroid points to the list
                initiate_list += 1
            for _, data in data_set.iterrows():  # Loops through the rows of the data set
                distance = None  # Initializes distance
                current_closest_point = []  # Keeps track of the current closes point
                iterator = 0
                for centroid in centroid_points:  # Loops through the k centroid points
                    euclid_distance = self.euclidean_distance(centroid,
                                                              data)  # Gets the distance between the centroid and the data point
                    if distance is None or euclid_distance < distance:  # Updates the distance to keep track of the closest point
                        distance = euclid_distance
                        current_closest_point = [data, iterator]
                    iterator += 1
                clusters[current_closest_point[1]].append(
                    list(current_closest_point[0]))  # Appends the list to the clusters for specific centroids
            for closest_cluster in clusters:  # Loops through the closest cluster list
                current_points.append(closest_cluster)
            centroid_points = self.get_new_cluster(
                current_points)  # Calls the get new cluster function to get the mean values and run through the updated centroid points
            print("Previous Clusters:")
            print(pd.DataFrame(previous_points))
            print("\nUpdated Clusters:")
            print(pd.DataFrame(centroid_points))

            if centroid_points == previous_points:
                print("\n----------------- K-Means has converged! -----------------")
                break

        return centroid_points

    def k_random_point(self, data_set, k_val):  # Method to grab k_random rows for centroid method
        print("\n-----------------Getting K Random Cluster Points-----------------")
        current_point = []  # List for current random point in loop
        centroid_points = []  # List of centroid points type Series
        for k in range(k_val):  # Grabs k Centroids
            length = len(data_set[1]) - 1  # Gets the length of the dataframe
            # Following row iteration with iteritems() sourced from https://stackoverflow.com/questions/28218698/how-to-iterate-over-columns-of-pandas-dataframe-to-run-regression/32558621 User: mdh and mmBs
            for col in data_set.iteritems():  # Loops through columns
                while True:  # While loop if random value is not found in column
                    random_int = random.randint(0, length)  # Selects a random row
                    try:
                        current_point.append(col[1][random_int])  # Appends the column point to the current point list
                        break
                    except:
                        pass

            centroid_points.append(current_point)  # Appends the point to a list to be returned
            current_point = []  # Resets current point

        return centroid_points  # Returns a Series of centroid points

    def get_new_cluster(self, current_clusters):  # Method to get the sum of values of the clusters
        print("\n----------------- Updating K-Means Clusters -----------------\n")
        mean_cluster = []  # Instantiates a list of the updated clusters
        for cluster in current_clusters:  # Loop through the current clusters to get the sum of the values
            current_point = []
            cluster_length = len(cluster)  # Passed to mean_current_cluster
            str_dict = {}  # Dictionary of the first column str labels
            for point in cluster:  # Loop through each data point in a cluster
                iterator = 0
                for index in point:
                    if type(index) is str:
                        try:
                            if index in str_dict.keys():
                                str_dict[index] += 1  # Increments the count of a particular string
                            else:
                                str_dict[index] = 1  # Instantiates a value for a particular string
                            current_point[iterator] = index  # Place holder in the list
                        except:
                            current_point.append(index)  # Place holder in the list
                    elif type(index) is np.float64 or type(index) is float:  # Handles float values
                        try:
                            current_point[iterator] = current_point[iterator] + float(
                                index)  # Sums the value for this particular column in the loop
                        except:
                            current_point.append(index)  # Instantiates a value for the index location

                    elif type(index) is int or type(index) is np.int64:  # Handles Int values
                        try:
                            current_point[iterator] += index  # Sums the value for this column
                        except:
                            current_point.append(index)  # Instantiates a value for this location in the list.
                    iterator += 1
            mean_cluster.append(self.mean_current_cluster(cluster_length, current_point,
                                                          str_dict))  # Appends the new cluster value to be returned.

        return mean_cluster

    def mean_current_cluster(self, cluster_length, current_point,
                             str_dict):  # This function does the math for the new centroid
        highest_char_count = 0  # Decided to use the highest occurring string.
        if str_dict.keys().__len__() > 0:
            iterator = 1
        else:
            iterator = 0
        for char in str_dict.keys():  # Loops through the string dictionary
            if str_dict[char] > highest_char_count:
                highest_char_count = str_dict[char]
                current_point[0] = char  # Sets the first location in the centroid list to the most occurring string.

        for index in range(iterator, len(current_point)):  # Loops through the values for the mean of the cluster
            current_point[
                index] /= cluster_length  # Divides the sum by the length of the columns in the cluster data set.
            if index == len(current_point) - 1:
                current_point[index] = int(
                    current_point[index])  # Last value in the data set is an INT. This is a type cast.
        return current_point

    def predict_centroids(self, centroids, data_set): # Method to return closest cluster to test data
        print("\n----------------- Predicting Closes Cluster on Test Data -----------------\n")

        for _, data in data_set[self.current_data_set].iterrows():  # Loops through the rows of the data set
            distance = None  # Initializes distance
            closest_centroid = None  # Keeps track of the current closes centroid cluster
            closest_centroid_euclidian_distance = None # Keeps track of the closest euclidian distance.
            cluster_val = 1
            for centroid in centroids:  # Loops through the k centroid points
                euclid_distance = self.euclidean_distance(centroid,
                                                          data)  # Gets the distance between the centroid and the data point

                if distance is None or euclid_distance < distance:  # Updates the distance to keep track of the closest point
                    distance = euclid_distance
                    # closest_centroid = centroid
                    closest_centroid = cluster_val
                    closest_centroid_euclidian_distance = distance
                cluster_val += 1
            # Print closest cluster to the test data point.
            print("\nEuclidian Distance to Closest K-Means Cluster: ", closest_centroid_euclidian_distance)
            print("Closest Cluster: Cluster ", closest_centroid )
