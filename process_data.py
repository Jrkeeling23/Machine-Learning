import pandas as pd
import numpy as np


class Data:
    """
    Anything relating to data such as imputations should be done here.
    """

    def __init__(self):
        """
        Keep the data and functions handling data private (denoted with __ )!
        Initializes data dictionary, which holds the data frames alongside there names.
        """
        self.data_dict = {}
        self.load_data()
        # TODO: split data will set test/train_dict equal to appropriate data.
        self.test_dict = None
        self.train_dict = None
        # TODO: Should we have dict for test and train data?

        if self.pre_process_data() is False:
            # TODO: complete the data
            pass
        else:
            pass  # Keep this pass

    def load_data(self):
        """
        Loads data into a dictionary
        :return: None
        """
        # Classification
        self.data_dict["abalone"] = pd.read_csv(r'data/abalone.data', header=None)
        self.data_dict["car"] = pd.read_csv(r'data/car.data', header=None)
        self.data_dict["segmentation"] = pd.read_csv(r'data/segmentation.data', header=None)
        # Regression
        self.data_dict["machine"] = pd.read_csv(r'data/machine.data', header=None)
        self.data_dict["forest_fires"] = pd.read_csv(r'data/forestfires.data', header=None)
        self.data_dict["wine"] = pd.read_csv(r'data/wine.data', header=None)

    def pre_process_data(self):
        """
        Check if data is complete. No missing values, etc...
        :return: Boolean TODO: Or just fix in here
        """
        # TODO: Ensure that the data is complete
        pass

    def split_data(self):
        """
        Split data for testing and training
        :return:
        """
        # TODO: Split data accordingly... Not sure the exact percent needed for test data
        pass

    def k_fold(self, k_val):
        """
        Use k-fold to split data
        TODO: 10 Fold or 5 if need be.
        :param k_val: k value to set size of folds.
        :return:
        """
        pass
