import pandas as pd
import numpy as np
import csv


class Data:
    """
    Anything relating to data such as imputations should be done here.
    """

    def __init__(self):
        """
        Keep the data and functions handling data private (denoted with __ )!
        Initializes data dictionary, which holds the data frames alongside there names.
        """
        # label = ['Age', ,
        self.data_dict = {}
        self.load_data()  # load raw data frames

        self.test_dict = {}
        self.train_dict = {}
        self.split_data()  # split data into testing and training sets



    def load_data(self):
        """
        Loads data into a dictionary
        :return: None
        """
        # Classification/home/justin/Desktop/ml_p2
        self.data_dict["abalone"] = pd.read_csv(r'data/abalone.data', header=None)
        self.data_dict["car"] = pd.read_csv(r'data/car.data',
                                            header=None)  # TODO figure out distance function for car data
        # TODO Load segmentation data
        self.data_dict["segmentation"] = pd.read_csv(r'data/segmentation.data', header=1, skiprows=[0])
        # Regression
        self.data_dict["machine"] = pd.read_csv(r'data/machine.data', header=None)
        self.data_dict["forest_fires"] = pd.read_csv(r'data/forestfires.data',
                                                     header=None)  # TODO figure out distance function for forest fires data
        self.data_dict["wine"] = pd.read_csv(r'data/wine.data',
                                             header=None)  # TODO Figure out distance function for wine data

    def pre_process_data(self):
        """
        Check if data is complete. No missing values, etc...
        :return: Boolean TODO: Or just fix in here
        """
        # TODO: Ensure that the data is complete
        # TODO: REFORMAT S.T LABEL IS ALWAYS LAST
        pass

    def split_data(self):
        """
        Split data for testing and training
        :return:
        """
        for data_set_name, data_set in self.data_dict.items():  # iterate through
            # use numpys split with pandas sample to randomly split the data
            training_data_temp, test_data_temp = np.split(data_set.sample(frac=1), [int(.8 * len(data_set))])
            # add training/testing data into dictionary with corresponding data set name
            if training_data_temp is not None and test_data_temp is not None:
                self.train_dict[data_set_name] = training_data_temp
                self.test_dict[data_set_name] = test_data_temp




    def split_k_fold(self, k_val, dataset):
        """
        Split data into list of K different parts
        :param k_val: k value to set size of folds.
        :return: list of lists where arranged as follows [[train,test], [train, test]] repeated k times
        where train is traing data (index 0) and test is testing data (index 1)
        """
        k__split_data = np.array_split(dataset, k_val) # splits dataset into k parts
        # now we need to split up data into 1 list and k others combined into 1 list for test/train
        test_train_sets = []
        temp_list = [None] * 2
        length = len(k__split_data)
        # create these new lists and add them to test_train_sets
        for i in range(length):  # go through every split list
            # APPARENTLY PYTHON DEVS THOUGHT IT WAS A GOOD FUCKING IDEA TO MAKE LISTS THAT HAVE DIFFERENT NAMES BOTH
            # REMOVE VALS WHEN THE REMOVE FUNCTION IS APPLIED TO ONE OF THEM.   WHY GOD WHY
            data_to_combine = np.array_split(dataset, k_val)
            temp_list[0] = k__split_data[i]
            del data_to_combine[i]
            temp_list[1] = pd.concat(data_to_combine)
            test_train_sets.append(temp_list)
            i += 1

        return test_train_sets

    def get_label_col(self, data_name):
        col_loc = {'abalone': 8, 'car': 5, 'segmentation': 0, 'machine': 0, 'forest_fires': 12, 'wine': 0}
        return col_loc[data_name]
