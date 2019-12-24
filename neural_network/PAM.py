import collections

from Cluster import KNN
import pandas as pd


class PAM(KNN):
    """
    Inheritance allows PAM to use functions in KNN, or override them, and use it class variables.
    """

    def __init__(self, k_val, data_instance):
        super().__init__(k_val, data_instance)
        self.current_medoids = pd.DataFrame().reindex_like(data_instance.train_df)
        self.current_medoid_indexes = []

    @staticmethod
    def assign_random_medoids(df, k):
        """
        randomly selects examples to represent the medoids
        :param df: data frame to get random points from
        :param k: number of medoids to instantiate
        :return: k number of medoids
        """
        medoid_list = []
        medoid_index = []
        rand_med = df.sample(n=k)
        for index, row in rand_med.iterrows():
            medoid_index.append(index)  # add to static variable; eases checking if data in medoids
            medoid_list.append(Medoid(row, index, pd.DataFrame(columns=df.columns, index=None)))
        return medoid_list, medoid_index

    def assign_data_to_medoids(self, df, medoid_list):
        """
        Assigns the remaining data points to medoids
        :param df: data to add to medoids
        :param medoid_list: list of medoids
        :return: None
        """
        Medoid.resets(medoid_list, df)
        for index, row in df.iterrows():  # iterate through all the data
            if index in self.current_medoid_indexes:  # do not assign a medoid to a medoid
                continue  # next index if a medoid
            list_of_tuples = self.find_closest_medoid(row, medoid_list)  # determines closest medoid and returns tuples
            med = list_of_tuples[0][0]  # closest medoid since ordered closest to furthest
            med.cost += list_of_tuples[0][1]  # distance from closest medoid
            med.encompasses.loc[index] = row  # append to the closest medoid point

    def find_closest_medoid(self, query, medoid_list):
        """
        Return the closest medoid to a given query point
        :param query: point to find closest medoid
        :param medoid_list: medoids to compare to
        :return: medoid and distance
        """
        distances = {}
        for med in medoid_list:
            distances[med] = super().get_euclidean_distance(query, med.row)
        return self.order_by_dict_values(distances)

    @staticmethod
    def order_by_dict_values(dictionary):
        """
        Orders least to greatest a given dictionary by value (not key) and returns a list of tuples [(key, val_min),
        ..., (key, val_max)]
        :param dictionary: dictionary to sort
        :return: an ordered list of tuples by value
        """
        return sorted(dictionary.items(), key=lambda item: item[1])

    def perform_pam(self):
        """
        Performs partition around medoids. Nothing else needs to be called by the user.
        1) Assign random medoids.
        2) REPEAT THE FOLLOWING UNTIL NO MORE SWAPS
            3) Assign Data to Medoids
            4) Iterate through all medoids
                5) Iterate through all non-medoids
                    6) Determine non-medoid is a better fit for initial medoids current encompasses data points

        :return: TODO: determine what needs to be returned: medoids?
        """
        self.current_medoids, self.current_medoid_indexes = self.assign_random_medoids(self.train_df, self.k)
        starting_medoids_index = self.current_medoid_indexes.copy()  # first iteration: for printing
        starting_medoids = self.current_medoids.copy()  # first iteration medoids
        start_distort = 0  # keep first iteration distortion for printing purposes
        print("\n\n_______________ Begin Finding Better Medoids _______________\n")
        print("Initial Medoid Indexes: ", self.current_medoid_indexes)  # prompt user of initial medoids
        compare = lambda x, y: collections.Counter(x) == collections.Counter(y)  # lambda expr to compare a collection
        checker = False  # used as a switch to catch oscillation
        temp1 = self.current_medoid_indexes  # list to check for oscillation
        temp2 = self.current_medoid_indexes  # list to check for oscillation
        temp_med1 = None  # temp medoids for oscillation purposes
        temp_med2 = None  # temp medoids for oscillation purposes
        first_iter = True
        while True:  # continue until oscillation or no more changes
            if checker:  # cache system
                temp1 = self.current_medoid_indexes
                temp_med1 = self.current_medoids.copy()
                checker = False
            else:  # cache system
                temp2 = self.current_medoid_indexes
                temp_med2 = self.current_medoids.copy()
                checker = True
            self.assign_data_to_medoids(self.train_df,
                                        self.current_medoids)  # at the beginning of every iteration, assign non medoids to medoids.
            if first_iter:  # first iteration
                for med in self.current_medoids:  # iterate through first medoids
                    start_distort += med.cost  # get all costs for distortion
                first_iter = False  # do not overwrite
            changed_list, indexes = self.medoid_swap(self.current_medoids.copy(),
                                                     self.train_df)  # for all medoids; test all data points to find better fit medoid
            if compare(temp1, indexes) or compare(temp2, indexes):  # check if oscillating or complete
                print("\n_______________ Found Best Fitting Medoids _______________\n")
                # BELOW: Returns best fit indexes, distortion, and the actual medoids
                return self.print_final(starting_medoids_index, start_distort, temp_med1, temp_med2, temp1, temp2)

            else:
                print("\nInitial Medoid list: ", self.current_medoid_indexes, "\nReturned Medoid List: ", indexes,
                      " Cache[0]: ", temp1, " Cache[1]: ", temp2)
                self.current_medoids = changed_list  # swaps to change medoids
                self.current_medoid_indexes = indexes  # swaps to change medoid indexes
                print("\n\n---------- Continue Finding Better Medoids ----------\n")

    def medoid_swap(self, medoid_list, df):
        """
        Swap out a medoid with a non medoid and compare
        :param medoid_list: current medoid list
        :param df: data set to use as medoids
        :return:
        """
        temp_medoid_list = medoid_list  # needs a reference to overwrite
        temp_indexes = self.current_medoid_indexes.copy()  # needs a reference to overwrite

        for med_index in range(len(temp_medoid_list)):  # iterate through all the medoids
            initial_medoid = medoid_list[med_index]  # reference for printing
            temporary_medoid = initial_medoid  # references the medoid being replaced (to potentially overwrite)
            print("\nCurrent Medoid being Swapped Out: ", initial_medoid.index)
            for index, row in df.iterrows():  # for each medoid: test a non medoid
                if index not in temp_indexes:  # ensure that it is not medoid
                    test_medoid = Medoid(row, index, temporary_medoid.encompasses)  # instantiate a new medoid
                    test_medoid_list = temp_medoid_list.copy()  # copy actual medoid_list
                    test_medoid_list[
                        med_index] = test_medoid  # replace actual medoid from temp list with testing medoid
                    test_medoid.cost += self.test_medoid_distortion(
                        test_medoid)  # get cost of medoid to all of its encompassed
                    test_distort, init_distort = self.calculate_distortions(test_medoid, temporary_medoid,
                                                                            # need initial and test medoid to remove or add its cost for more accurate distortion values
                                                                            test_medoid_list,
                                                                            # need to add initial medoid to cost
                                                                            temp_medoid_list)  # need to get remove test medoid cost
                    if init_distort > test_distort and test_distort is not 0:  # if initial cost is greater than (ensure it is not considering itself... 0)
                        print(
                            "SWAPPING JUSTIFICATION: Initial Distortion = %s  Test Distortion: %s" % (
                            init_distort, test_distort))
                        print("\t\tSwap Initial Medoid %s with Test Medoid %s" % (
                        temporary_medoid.index, test_medoid.index))
                        # Below: swap the values for the next "test" medoid so that it is compared to current test medoid
                        temporary_medoid = test_medoid
                        temp_medoid_list = test_medoid_list
                        temp_indexes[med_index] = test_medoid.index
                    else:
                        continue  # next potential "better fit" medoid
        return temp_medoid_list, temp_indexes

    def test_medoid_distortion(self, test_medoid):
        """
        Calculate a single medoids distortion from its encompassed
        :param test_medoid: test medoid to calculate
        :return: float, distortion
        """
        distortion = 0
        for index, query in test_medoid.encompasses.iterrows():
            distortion += self.get_euclidean_distance(query, test_medoid.row)
        return distortion

    def calculate_distortions(self, test_medoid, init_medoid, test_medoid_list, init_medoid_list):
        """
        Created the distortions by addind the initial medoids cost and removing the test_medoid cost from the original.
        :param test_medoid: potential new medoid
        :param init_medoid: medoid possibly to be swapped out
        :param test_medoid_list: medoid list with test med and no initial med
        :param init_medoid_list: medoid list  with init med and not test med
        :return: test and initial distortions.
        """
        initial_distortion = 0
        for med in init_medoid_list:  # calculate initial distortion
            initial_distortion += med.cost
        test_medoid.cost = self.test_medoid_distortion(test_medoid)  # assign cost to test medoid
        test_medoid_distortion = 0  # instantiate partial distortion
        for med in test_medoid_list:
            test_medoid_distortion += med.cost  # obtain partial test distortion
        init_medoid_assigned_to = self.find_closest_medoid(init_medoid.row, test_medoid_list)  # obtain medoid and cost
        init_cost = init_medoid_assigned_to[0][1]  # additional cost to consider
        test_medoid_assigned_to = self.find_closest_medoid(test_medoid.row,
                                                           init_medoid_list)  # obtain test cost to redact
        test_cost = test_medoid_assigned_to[0][1]  # subtract cost for new distortion
        tested_distortion = test_medoid_distortion - test_cost + init_cost
        return tested_distortion, initial_distortion

    def print_final(self, start_indexes, start_distortion, temp_med1, temp_med2, temp1, temp2):
        """
        Determines better value for oscillation indexes as well as prints the results
        :param temp2: oscillating indexes
        :param temp1: oscillating indexes
        :param start_indexes: first iteration indexes
        :param start_distortion: first iteration distortion
        :param temp_med1: oscillating medoids
        :param temp_med2: oscillating medoids
        :return: best fitting data
        """
        dis1 = 0
        dis2 = 0
        # below for loops and if-else choose best out of the oscillation points
        for med in temp_med1:
            dis1 += self.test_medoid_distortion(med)  # get distortion of oscillating medoid
        for med in temp_med2:
            dis2 += self.test_medoid_distortion(med)  # get distortion of oscillating medoid
        if dis1 > dis2:  # compare costs and assign the one to be best choice
            best_distortion = dis2
            best_indexes = temp2
            best_medoids = temp_med2
        else:
            best_distortion = dis1
            best_indexes = temp1
            best_medoids = temp_med1
        # prints the change in the medoids
        print("First Iteration Medoid Indexes: ", start_indexes, "\t\tDistortion: %s" % start_distortion)
        print("Final Iteration Medoid Indexes: ", best_indexes, "\t\tDistortion: %s" % best_distortion)
        return best_indexes, best_distortion, best_medoids


class Medoid:

    def __init__(self, row, index, encompasses):
        """
        Medoids instance. Initializes the medoids current point, what data is contained within that medoid and an index
        :param row: the point representing the medoid
        :param index: index of the point
        """
        self.row = row
        self.encompasses = encompasses  # Dataframe of its medoids
        self.index = index  # index of data frame
        self.cost = 0  # individual medoid cost
        self.best_medoid = None  # list of test_medoids to potentially

    def get_medoid_encompasses(self):
        """
        Getter function to use the encompassed list
        :return: the data points the medoid encompasses
        """
        return self.encompasses

    @staticmethod
    def resets(medoid_list, df):
        """
        reset the costs when recalculating the cost of medoids
        :return: None
        """
        for medoid in medoid_list:
            medoid.cost = 0
            medoid.encompasses = pd.DataFrame(columns=df.columns, index=None)
