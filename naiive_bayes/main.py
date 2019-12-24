"""
CSCI-447 Machine Learning
Justin Keeling, Alex Harry, Andrew Smith, John Lambrect
11 Oct 2019
"""
from process_data import Data
from KNN import KNN


def main():
    """
    Calls function in other files until program is finished.
    :return: None
    """
    knn = KNN()
    data = Data()  # loads the data and checks if complete

    while True:
        data.split_data()  # split into both test and train
        predicted_class = {}  # holds data_set_name and a list of predicted classes

        for key, train_data_set in data.train_dict:  # iterate through data and get key(Data name) and data_set
            print("Current Data Set: ", key)
            predicted_class[key] = []  # create a list of for a data set of predicted values
            test_data_set = data.test_dict[key]  # TODO: Use same keys for all dictionaries; Access testing data by key.
            for _, query_point in train_data_set.iterrows():
                # give query example and its corresponding train_data_set, along with # of desired neighbors to consider
                predicted_class[key].append(knn.perform_knn(query_point, train_data_set, 5))

