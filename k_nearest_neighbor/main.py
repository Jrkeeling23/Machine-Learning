"""
CSCI-447 Machine Learning
Justin Keeling, Alex Harry, Andrew Smith, John Lambrect
11 Oct 2019
"""
from process_data import Data
from KNN import KNN
from loss_functions import LF
from medoids import KMedoids

# def run_knn():
#     """
#     Calls function in other files until program is finished.
#     :return: None
#     """
#     knn = KNN()
#     data = Data()  # loads the data and checks if complete
#
#     while True:
#         data.load_data()
#         data.split_data()  # split into both test and train
#         predicted_class = {}  # holds data_set_name and a list of predicted classes
#
#         for name, train_data_set in data.train_dict.items():  # iterate through data and get key(Data name) and data_set
#             print("Current Data Set: ", name)
#             predicted_class[name] = []  # create a list of for a data set of predicted values
#             test_data_set = data.test_dict[name]  # TODO: Use same keys for all dictionaries; Access testing data by key.
#             for _, query_point in train_data_set.iterrows():
#                 # give query example and its corresponding train_data_set, along with # of desired neighbors to consider
#                 predicted_class[name].append(knn.perform_knn(query_point, train_data_set, 5, name, data))

knn = KNN()
data = Data()  # loads the data and checks if complete
lf = LF()
data.load_data()


def run_zero_loss():
    """
    Calls function in other files until program is finished.
    :return: None
    """
    data.split_data()  # split into both test and train
    lf.zero_one_loss(data.test_dict['abalone'].sample(n=400), 5, 'abalone', data)


def run_k_means(indata):  # Run k-means on wine data set'knn = KNN()
    knn = KNN()
    data = Data()  # loads the data and checks if complete
    data.split_data()
    in_data = {'abalone':indata}
    knn.data = data
    knn.current_data_set = 'abalone'  # Set the data set to be used to wine
    centroids = knn.centroids(in_data, 5)  # Get the k-means clusters
    knn.predict_centroids(centroids, data.test_dict)  # Predict the closest cluster


def run_condense():
    k_val = 5
    return knn.condense_data(data.test_dict['abalone'].sample(n=400), k_val, 'abalone', data)


def run_edited():
    pass


def run_medoids(test_data, train):
    md = KMedoids(test_data, train)
    predict = md.perform_medoids(3, 'abalone')
    lf.mean_squared_error(predict, test_data)


if __name__ == "__main__":
    # run_knn()
    # run_zero_loss()
    # run_k_means()
    condensed_data = run_condense()
    # edited_data = run_edited()
    run_k_means(condensed_data)
    run_medoids(data.test_dict['abalone'], condensed_data)
