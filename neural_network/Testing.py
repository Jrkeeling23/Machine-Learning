import unittest
from loss_functions import LF
from KMeans import Kmeans
from PAM import PAM
from Data import Data, DataConverter
import pandas as pd
import numpy as np
from Cluster import KNN
from NeuralNetwork import NeuralNetwork, NetworkClient
import collections
from RBFNet import RBFReg
from RBFNet import RBFClass
from loss_functions import LF
from RBFNetKMean import RBFRegK


class MyTestCase(unittest.TestCase):

    def test_data_conversion_to_original(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)
        df = data.df.sample(n=50)
        data.split_data(data_frame=df)
        converter = DataConverter()
        converted = converter.convert_data_to_original(data.train_df)
        mismatch = False
        dt = converter.convert_data_to_original(data.train_df.copy())
        for convert in converted.values:
            if convert not in dt.values:
                mismatch = True
        self.assertFalse(mismatch)

    def test_determine_closest_in_dictionary(self):
        """
        Test for PAM function
        """
        test_dict = {  # arbitrary set dictionary to test
            "1": 55, "2": 22, "3": 11, "4": 1
        }
        result_list = PAM.order_by_dict_values(test_dict)  # returns list of tuples (not dictionary)
        result = result_list[0][1]  # obtain first value dictionary is ordered by
        self.assertEqual(result, 1, "minimum value")  # determines if it is smallest element
        self.assertNotEqual(result, 55, "Not Max Val")

    def test_PAM_super(self):
        """
        Test if inheritance is working properly
        test if euclidean is working properly for both single and dictionary returns
        """
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=10)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        pam = PAM(k_val=2, data_instance=data)  # create PAM instance to check super
        pam_train_data = pam.train_df  # train_df is an instance from parent class
        self.assertTrue(pam_train_data.equals(data.train_df), "Same data")
        self.assertFalse(pam_train_data.equals(data.test_df))

        row_c, row_q = np.split(pam_train_data, 2)  # split the same data into size of
        _, row_comp = next(row_c.copy().iterrows())  # get a row
        _, row_query = next(row_q.copy().iterrows())  # get a row
        dict_dist = pam.get_euclidean_distance_dict(row_query, row_c)

        single_distance = pam.get_euclidean_distance(row_query, row_comp)  # get distance
        self.assertTrue(isinstance(single_distance, float))  # check the it returns a float
        self.assertTrue(isinstance(dict_dist, dict))  # check if it is a dictionary

    def test_medoid_swapping(self):
        """
        Just run to see values being swapped
        :return:
        """
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=300)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        pam = PAM(k_val=3, data_instance=data)  # create PAM instance to check super
        index, distort, medoids = pam.perform_pam()

    def test_KNN(self):
        """
        Test if KNN is returning a class
        :return:
        """
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=10)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        k_val = 5
        knn = KNN(k_val, data)
        nearest = knn.perform_KNN(k_val, df.iloc[1], data.train_df)
        print(nearest)

    def test_euclidean(self):
        """
        Test if euclidean distance is working
        :return:
        """
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=10)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        knn = KNN(5, data)
        print(knn.get_euclidean_distance(df.iloc[1], df.iloc[2]))

    def test_edit(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False)
        df = data.df.sample(n=50)
        data.split_data(data_frame=df)
        knn = KNN(5, data)
        knn.edit_data(data.train_df, 5, data.test_df, data.label_col)

    def test_data_conversion_to_numerical(self):
        data = Data('segmentation', pd.read_csv(r'data/segmentation.data', header=None, skiprows=4), 8)
        df = data.df.sample(n=209)
        data.split_data(data_frame=df)

    def test_k_means(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=200)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        k_val = 5
        knn = KNN(k_val, data)
        kmeans = Kmeans(k_val, data)
        clusters = kmeans.k_means(data.train_df, k_val)
        converter = DataConverter()
        dt = converter.convert_data_to_original(data.train_df.copy())
        mismatch = False
        for cluster in clusters.values:
            if cluster not in dt.values:
                mismatch = True
        self.assertFalse(mismatch)

    def test_layers(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False)
        df = data.df.sample(n=4)
        data.split_data(data_frame=df)
        network = NeuralNetwork(data_instance=data)
        network.make_layers(2, 4)

    def test_mse(self):
        lf = LF()
        data = [1, 3, 2, 3, 4, 4, 3, 2, 3, 4, 3, 3]
        label = [1, 5, 2, 1, 6, 5, 3, 2, 3, 2, 3, 3]
        self.assertEquals(round(lf.mean_squared_error(data, label), 9), 1.416666667)

    def test_zero_one(self):
        lf = LF()
        data = [1, 3, 2, 3, 4, 4, 3, 2, 3, 4, 3, 3]
        label = [1, 5, 2, 1, 6, 5, 3, 2, 3, 2, 3, 3]
        self.assertEquals(round(lf.zero_one_loss(data, label), 10), 0.4166666667)

    def test_knn_condensed(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=350)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        cluster_obj = KNN(5, data)
        condensed_data = cluster_obj.condense_data(data.train_df)
        size_after = condensed_data.shape[0]
        size_prior = data.train_df.shape[0]
        self.assertGreater(size_prior, size_after)

    def test_discretize(self):
        data = Data('segmentation', pd.read_csv(r'data/segmentation.data', header=None, skiprows=4), 8, True)
        data.regression_data_bins(4, quartile=True)
        data.regression_data_bins(4, quartile=False)

    def test_network_prediction(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False)
        df = data.df.sample(n=10)
        data.split_data(data_frame=df)
        network = NeuralNetwork(data_instance=data)
        layers, output_set = network.make_layers(2, 6)
        output_prediction = network.sigmoid(layers, df.iloc[0].drop(data.label_col))
        print(network.prediction(output_set, output_prediction))

    def test_cost(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False)
        df = data.df.sample(n=10)
        data.split_data(data_frame=df)
        network = NeuralNetwork(data_instance=data)
        layers, output_set = network.make_layers(2, 6)
        output_prediction = network.sigmoid(layers, df.iloc[0].drop(data.label_col))
        print(network.cost(output_prediction, output_set, df.iloc[0][data.label_col]))

    def test_backprop(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False)
        df = data.df.sample(n=50)
        data.split_data(data_frame=df)
        network = NeuralNetwork(data_instance=data)
        layers, output_set = network.make_layers(1, 4)
        output_predictions = []
        costs = []
        for index, row in data.train_df.iterrows():
            output_predictions.append(network.sigmoid(layers, row.drop(data.label_col)))
            costs.append(network.cost(output_predictions[-1], output_set, row[data.label_col]))

    def test_it_all(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False)
        df = data.df.sample(n=200)
        data.split_data(data_frame=df)
        client = NetworkClient(data)
        layers, outputset, network = client.train_it(1, 10, .3, .5, 15)
        # print(client.testing(layers, outputset, network))  # prints total
        lf = LF()
        pred, actual = client.testing(layers, outputset, network)
        print("Predicted Set, ", pred, " Actual Set: ", actual)

    def test_rbfReg(self):
            #data = Data('winequality-white', pd.read_csv('data/winequality-white.csv', header=None), 8)  # load data
            data = Data('winequality-white', pd.read_csv('data/winequality-white.csv', header=None), 11)  # load data
            df = data.df.sample(n=100)  # minimal data frame

            test_df = pd.read_csv('data/winequality-white.csv', header=None)
            test2_df = test_df.iloc[:101, :]
            #print(test2_df[11])
            print("Checking DF set")
            print(df[df.columns[-1]])

            cols = df.columns
            for col in cols:
                df[col] = df[col].astype(float)
            expected = df[df.columns[-1]]

            #print(expected[1])

            df = df.iloc[:, :-1]
            test2_df = test2_df.iloc[:, :-1]
            data.split_data(data_frame=df)  # sets test and train data
            # will have high error due to small dataset, but just a test to show how this works
            rbf = RBFReg(clusters=12, maxruns=1000)

            rbf.trainReg(data.train_df, expected, data)

            predicts = rbf.predictReg(data.test_df, data)
            expc_list = expected.values.tolist()
            print("predicts")
            print(predicts)
            print("expected")
            print(expc_list)

            print("MSE")
            mse = rbf.mean_squared_error(predicts, expc_list)
            print(mse)


    def test_rbfClass(self):
            #data = Data('winequality-white', pd.read_csv('data/winequality-white.csv', header=None), 8)  # load data
            data = Data('abalone', pd.read_csv('data/abalone.data', header=None), 11)  # load data
            df = data.df.sample(n=100)  # minimal data frame

            cols = df.columns
            for col in cols:
                df[col] = df[col].astype(float)
            data.split_data(data_frame=df)
            expected = data.test_df[data.test_df.columns[-1]]
            #data. = df.iloc[:, :-1]
              # sets test and train data
            # print(data.test_df)
            # print(expected)
            # will have high error due to small dataset, but just a test to show how this works
            class_vals = list(range(1, 29))
            rbf = RBFClass(clusters=12, maxruns=400, out_nodes=len(class_vals))

            rbf.train(data, data.train_df, class_vals)

            predicts = rbf.predictClass(data.test_df, data)
            expc_list = expected.values.tolist()
            print("predicts")
            print(predicts)
            print("expected")
            print(expc_list)

            accuracy = rbf.zero_one_loss(predicts, expc_list)
    def test_edit_vs_condese(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)
        df = data.df.sample(n=350)
        data.split_data(data_frame=df)
        knn = KNN(5, data)
        edit = knn.edit_data(data.train_df, 5, data.test_df, data.label_col)
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8)  # load data
        df = data.df.sample(n=350)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        cluster_obj = KNN(5, data)
        condensed_data = cluster_obj.condense_data(data.train_df)
        size_after = condensed_data.shape[0]
        print("----------")
        print(edit.shape[0])
        print(size_after)
        if size_after < edit.shape[0]:
            print("Run condensed")
        else:
            print("Run edited")


    def test_rbfRegKMeans(self):
            #data = Data('winequality-white', pd.read_csv('data/winequality-white.csv', header=None), 8)  # load data
            data = Data('winequality-white', pd.read_csv('data/winequality-white.csv', header=None), 11)  # load data
            df = data.df.sample(n=100)  # minimal data frame

            test_df = pd.read_csv('data/winequality-white.csv', header=None)
            test2_df = test_df.iloc[:101, :]
            #print(test2_df[11])
            print("Checking DF set")
            print(df[df.columns[-1]])

            cols = df.columns
            for col in cols:
                df[col] = df[col].astype(float)
            expected = df[df.columns[-1]]

            #print(expected[1])

            df = df.iloc[:, :-1]
            test2_df = test2_df.iloc[:, :-1]
            data.split_data(data_frame=df)  # sets test and train data
            # will have high error due to small dataset, but just a test to show how this works
            rbf = RBFRegK(clusters=4, maxruns=200)

            rbf.trainReg(data.train_df, expected, data)

            predicts = rbf.predictReg(data.test_df, data)
            expc_list = expected.values.tolist()
            print("predicts")
            print(predicts)
            print("expected")
            print(expc_list)

            print("MSE")
            mse = rbf.mean_squared_error(predicts, expc_list)
            print(mse)

    def test_runBothRBF(self):
        # data = Data('winequality-white', pd.read_csv('data/winequality-white.csv', header=None), 8)  # load data
        data = Data('winequality-white', pd.read_csv('data/winequality-white.csv', header=None), 11)  # load data
        df = data.df.sample(n=100)  # minimal data frame

        # print(test2_df[11])
        print("Checking DF set")
        print(df[df.columns[-1]])

        cols = df.columns
        for col in cols:
            df[col] = df[col].astype(float)
        expected = df[df.columns[-1]]

        # print(expected[1])

        df = df.iloc[:, :-1]

        data.split_data(data_frame=df)  # sets test and train data
        # will have high error due to small dataset, but just a test to show how this works
        rbf = RBFRegK(clusters=4, maxruns=200)
        rbf2 = RBFReg(clusters=4, maxruns=200)
        rbf.trainReg(data.train_df, expected, data)
        rbf2.trainReg(data.train_df, expected, data)

        predicts = rbf.predictReg(data.test_df, data)
        predictsmeds = rbf2.predictReg(data.test_df, data)
        expc_list = expected.values.tolist()
        print("predicts means")
        print(predicts)
        print("predicts medoids")
        print(predictsmeds)
        print("expected")
        print(expc_list)

        print("MSE Means")
        mse = rbf.mean_squared_error(predicts, expc_list)
        print(mse)

        print("MSE Medoids")
        mse2 = rbf2.mean_squared_error(predictsmeds, expc_list)
        print(mse2)




if __name__ == '__main__':
    unittest.main()
