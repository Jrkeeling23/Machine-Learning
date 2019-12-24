import pandas as pd
import numpy as np
from process_data import Data


class KNN:
    """
    Anything to do with k-nearest neighbor should be in here.
    """

    def perform_knn(self, query_point, train_data, k_val):
        """
        Function performs KNN to classify predicted class.
        :param k_val: number of neighbors
        :param query_point: all data to compare an example from test_data too.
        :param train_data:  all data to "query" and predict
        :return: Predicted class
        """
        distance_list = []
        for index, row in train_data.iterrows():  # iterate through all data and get distances
            distance_list.append(self.euclidean_distance(query_point, row, k_val))  # all features of an example to function
        return self.predict_by_distance(distance_list.sort(reverse=True))  # Predict by closest neighbors

    def euclidean_distance(self, query_point, comparison_point, k_val):
        """
        With multi dimensions: sqrt((x2-x1)+(y2-y1)+(z2-z1)+...))
        :param query_point: Testing example.
        :param comparison_point: example in training data.
        :return: float distance
        """
        feature_vector = []
        for feature_col in range(len(query_point)):
            # TODO: Iterate through each feature within points and find distance
            pass

    def predict_by_distance(self, distance_list):
        """
        Determines the prediction of class by closest neighbors.
        :param distance_list:
        :return: Predicted class
        """
        # TODO: Determine class
        pass

    def edit_data(self):
        """
        Edit values for edit_knn by classifying x_initial; if wrong, remove x_initial. (option1)
        OR... if correct remove (option 2)
        :return: Edited values back to KNN
        """
        # TODO: edit data according to pseudo code from class on 9/23
        pass

    def condense_data(self):
        """
        Condense the data set by instantiating a Z = None. Add x_initial to Z if initial class(x_initial) != class(x)
        where x is an example in Z.
        So: Eliminates redundant data.
        :return:
        """
        # TODO: edit data according to pseudo code from class on 9/23
        pass
