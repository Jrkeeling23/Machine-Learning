import unittest
from KNN import KNN
import pandas as pd
from process_data import Data
from loss_functions import LF
import numpy as np
from medoids import KMedoids


# Source to understand how to test in python: https://pymbook.readthedocs.io/en/latest/testing.html and https://docs.python.org/2/library/unittest.html
class Test(unittest.TestCase):

    def test_predict_by_distance(self):
        knn = KNN()
        class_list = ['right_pick','wrong_pick', 'wrong', 'right_pick', 'wrong_again']
        self.assertEqual(knn.predict_by_distance(class_list), 'right_pick')

    def test_predict_by_distance_with_conflict(self):
        knn = KNN()
        class_list = [17, 9, 17, 13, 9]
        self.assertEqual(knn.predict_by_distance(class_list), 17)

    def test_eucldean_distance(self):
        '''
        test to see of creating correct distances
        :return:
        '''
        query_point = [0, 0]
        comparison_point = [2, 2]
        knn = KNN()
        self.assertEqual(knn.euclidean_distance(query_point, comparison_point), (8**0.5))

    def test_perform_knn(self):
        query_point = [0, 0]
        comparison_point = [2, 2]
        knn = KNN()
        label = 'true'

    def test_condense_data(self):
        # compare that the size of  output pandas data frame is less than input (that CNN reduced the data)
        # importing part of abalone data to test this as we need the 2D structure
        knn = KNN()
        data = Data()
        data_temp = pd.read_csv(r'data/abalone.data', header=None)
        data_set = data_temp.loc[:400][:]  # get first 100 rows of data_set
        k_val = 5
        name = 'abalone'  # used in KNN, needed here
        cond_data = knn.condense_data(data_set, k_val, name, data)

        self.assertGreater(len(data_set.index),len(cond_data.index))

    def test_zero_one_loss(self):
        knn = KNN()
        lf = LF()
        data = Data()
        data_temp = pd.read_csv(r'data/abalone.data', header=None)
        data_set = data_temp.loc[:1000][:]  # get first 100 rows of data_set
        k_val = 5
        name = 'abalone'  # used in KNN, needed here
        #cond_data = knn.condense_data(data_set, k_val, name, data)
        self.assertIsNotNone(lf.zero_one_loss(data_set, k_val, name, data))

    def test_k_fold(self):
        data = Data()
        data_temp = pd.read_csv(r'data/abalone.data', header=None)
        data_split = data.split_k_fold(5, data_temp) #  split into 10 dif parts
        self.assertIs(len(data_split), 5) # check split into 2 groups
        self.assertIs(len(data_split[0]), 2) # check that it split into test and train

    def test_centroids(self):
        print("Testing Centroid")
        knn = KNN()
        data = Data()
        data.split_data()
        knn.data = data
        knn.current_data_set = 'wine'  # used in KNN, needed here
        centroids = knn.centroids(data.train_dict, 4)
        knn.predict_centroids(centroids, data.test_dict)
        print("End Centroid Test")

# Source to understand how to test in python: https://pymbook.readthedocs.io/en/latest/testing.html and https://docs.python.org/2/library/unittest.html
if __name__ == '__main__':
    unittest.main()
