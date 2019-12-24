import numpy as np
import pandas as pd
import random
from process_data import Data
from KNN import KNN


def get_label_col(data_name):
    col_loc = {'abalone': 8, 'car': 5, 'segmentation': 0, 'machine': 0, 'forest_fires': 12, 'wine': 0}
    return col_loc[data_name]


def euclidean_distance(query_point, comparison_point, data_name):
    """
    Performs the Euclidean distance function
    :param data_name:
    :param query_point: a data point
    :param comparison_point: a comparison point
    :param df: used to get the label columns
    :return: a distance
    """
    temp_add = 0  # (x2-x1)^2 + (y2 - y1)^2 ; addition part
    for feature_col in range(len(query_point)):
        if get_label_col(data_name) is feature_col:
            continue
        if type(query_point[feature_col]) is float or type(query_point[feature_col]) is int:
            temp_sub = (query_point[feature_col] - comparison_point[feature_col]) ** 2  # x2 -x1 and square
            temp_add += temp_sub  # continuously add until square root

    return temp_add ** (1 / 2)  # square root ... return the specific distance


class KMedoids:

    def __init__(self, test, train):
        self.df = train
        self.medoids_list = None  # contains a list of the Medoid instances
        self.data_name = None
        self.test = test
        self.temp_med_list = None
        self.current_med = None

    def perform_medoids(self, k, data_name):
        """
        Will carry out the all the medoid functions
        :param k: number of medoids
        :param data_name:
        :return:
        """
        self.data_name = data_name  # assigns the current data set being used for label column purposes
        self.select_random(k)  # select random data points to represent the medoids
        decreasing = True
        temp = 0
        while decreasing:
            print("\n\t\tREPEAT")
            decreasing = self.find_best_fit()
            self.reset()
            if self.temp_med_list == self.medoids_list and temp > 4:

                break
            elif self.temp_med_list == self.medoids_list and temp < 4:
                temp += 1
        return self.predict(k=3)

    def select_random(self, k):
        """
        randomly selects examples to represent the medoids
        :param k: number of medoids
        :return: k random data points
        """
        self.create_medoid_instances(self.df.sample(n=k))

    def assign_to_medoids(self, medoids_list):
        """
        assigns the remaining data points to its closest medoid
        :return:
        """
        for index, row in self.df.iterrows():
            if self.check_index(index, self.medoids_list, t_index=None):
                continue  # exclude to data points that are medoids
            else:
                distance_dict = self.check_all_medoid_distances(row,
                                                                medoids_list)  # check the row to medoids for
                # (continued paragraph) closest distance
                chosen_medoid, distance = self.sort_dict_by_value(distance_dict)

                chosen_medoid.assign_to_medoid(index, distance, row)  # assign the index of the data point to the medoid

    def check_all_medoid_distances(self, query_point, medoids_list):
        """
        calculate the Euclidean distances from the query point to the medoid
        :param query_point:
        :return:  distance dictionary
        """
        distances = {}  # dict to hold the indexes and the distances
        for med in medoids_list:
            distances[med] = euclidean_distance(query_point, med.medoid_point, self.data_name)
        return distances

    def sort_dict_by_value(self, sort_this):
        """
        determines the closest medoid to the query point
        :param sort_this: dictionary of distances from query point to medoids
        :return: closest medoid
        """

        first_iteration = True
        closest_medoid = None
        saved_closest = None

        for k, v in sort_this.items():
            if first_iteration:
                saved_closest = v  # grab specific medoid
                first_iteration = False  # assign first distance
                closest_medoid = k
                continue
            else:
                if saved_closest > v:
                    saved_closest = v  # change closest var to closer
                    closest_medoid = k
        return closest_medoid, saved_closest  # closest medoid and distance

    def create_medoid_instances(self, medoids):
        """
        creates Medoid instances from the randomly selected medoids
        :param medoids:
        :return:
        """
        self.medoids_list = []
        for index, row in medoids.iterrows():
            self.medoids_list.append(Medoids(row, index))  # initialize a new medoid and assign to the medoids list
        print("_________Initialize Medoids_________")

    def swap(self, medoid, temp_medoid, df_name):
        """
        finds cost of a possible medoid
        :param temp_medoid: point from df not in medoids list
        :param df_name: the name of data set for euclidean distance parameter
        :param df: the actual data set to get row from encompassed points
        :return: Boolean to infer a swap to caller function
        """
        for k, v in medoid.encompasses.items():  # iterate through all encompassed medoids
            if self.check_index(k, self.medoids_list, temp_medoid.index):  # if index is medoid or is a temp medoid
                continue
            else:
                distance = euclidean_distance(v, temp_medoid.medoid_point, df_name)  # get the distance of the point
                temp_medoid.cost += distance  # update the temporary medoids cost for distortion

        if temp_medoid.cost < medoid.cost and temp_medoid.cost is not 0:  # if the temp medoid is a better fit, SWAP!
            print("swapping medoid with cost ", medoid.cost, " with the medoid whose cost is ", temp_medoid.cost)
            medoid.index = temp_medoid.index
            medoid.cost = temp_medoid.cost
            medoid.medoid_point = temp_medoid.medoid_point

            return True
        else:
            # print("NOT swapping medoid with cost ", medoid.cost, " with the medoid whose cost is ", temp_medoid.cost)
            return False

    def check_index(self, index, medoids_list, t_index):
        temp = medoids_list.copy()
        if self.current_med is not None:
            medoids_list.append(self.current_med)
        for med in medoids_list:
            if t_index is not None:
                self.medoids_list = temp
                if index == med.index or index == t_index:
                    return True
            else:
                if index == med.index:
                    self.medoids_list = temp
                    return True
        self.medoids_list = temp
        return False

    def print_medoids(self):
        string = '\nMedoids List: '
        self.temp_med_list = self.medoids_list
        for med in self.medoids_list:
            string += str(med.index) + ", "
        print(string)

    def find_best_fit(self):
        decreasing = False
        self.print_medoids()
        self.assign_to_medoids(self.medoids_list)  # assign the remaining data points to its closest medoid
        for med in self.medoids_list:  # iterate through medoids
            print("\nCurrent medoid Index that is being updated ", med.index)
            self.current_med = med
            for index, row in self.df.iterrows():  # iterate though every point in the data set
                if self.check_index(index, self.medoids_list, t_index=None):  # if index is a medoid
                    continue
                else:  # not a medoid
                    temp_med = Medoids(row, index)  # create a temporary medoid
                    swap = self.swap(med, temp_med, self.data_name)  # swap the medoids
                    if swap is True:  # if swap took place, perform swap!
                        decreasing = True
                    else:
                        continue
        return decreasing

    def predict(self, k):
        print("----------Predicting----------")
        predict_list = []

        for index, row in self.test.iterrows():
            predict_list.append(self.find_k_closest_neighbors(row, k))
        print(predict_list)
        return predict_list

    def find_k_closest_neighbors(self, query, k):
        closest = None
        distance_dict = {}
        index_dict = {}
        for index, row in self.df.iterrows():
            distance_dict[index] = euclidean_distance(query, row, self.data_name)

        count = 0
        for key, value in sorted(distance_dict.items(), key=lambda item: item[1]):
            if count > k:
                break
            else:
                index_dict[key] = value
                count += 1
        meds_dict = self.get_clostest_med(index_dict)
        pred = self.predict_k_mediods(meds_dict)
        return pred

    def get_clostest_med(self, index_dict):
        meds_dict = {}
        for j in list(index_dict.keys()):

            for med in self.medoids_list:
                if j in list(med.encompasses.keys()):
                    meds_dict[j] = med
        return meds_dict

    def predict_k_mediods(self, distance_dict):
        freq_dict = {}
        first_itertion = True
        for dist, i in distance_dict.items():
            if not first_itertion:
                if i in freq_dict.values():
                    freq_dict[i] += 1
                else:
                    freq_dict[i] = 1
            else:
                freq_dict[i] = 1

        for key, value in sorted(freq_dict.items(), key=lambda item: item[1]):
            return key.index

    def reset(self):
        for med in self.medoids_list:
            med.reset_cost()


class Medoids:

    def __init__(self, point, index):
        """
        Medoids instance. Initializes the medoids current point, what data is contained within that medoid and an index
        :param point: the point representing the medoid
        :param index: index of the point
        """
        self.medoid_point = point
        self.encompasses = {}
        self.index = index
        self.cost = 0  # individual medoid cost

    def assign_to_medoid(self, index, distance, row):
        """
        Assigns data to the medoid
        :param index: index of data
        :return:
        """
        self.encompasses[index] = row
        self.cost += distance

    def reset_cost(self):
        """
        reset the costs when recalculating the cost of medoids
        :return:
        """
        self.cost = 0
