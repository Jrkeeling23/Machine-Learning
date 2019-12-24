import numpy as np
from PAM import PAM
from Cluster import KNN
import math
from KMeans import Kmeans

class RBFRegK:
    """
          Class to function as an RBF regression network (one output node)
          :param clusters, # of clusters (hidden nodes)
          :param isReg,  bool to check if regression or not
          :param learning_rate,  tuning paramater for our RBF net
          :param maxruns, maximum amount of cycles we want the RBF to run for


           """

    def __init__(self,  clusters,  maxruns=1000, learning_rate=.01, ):

        self.clusters = clusters
        self.learning_rate = learning_rate
        # weight array,  use out_nodes size 1  for reg
        #self.weights = np.random.uniform(-self.learning_rate, self.learning_rate, size=self.clusters)
        self.weights = np.random.randn(clusters)
        # bias term, vector of size out_nodes (so size 1 for reg as 1 output node)
        self.bias = np.random.randn(1)
        self.maxruns = maxruns
        self.std = None
        self.medoids = None


    def gaus(self, x, c, s):
            """ gausian"""
            return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)

    def calcHiddenOutputs(self, input, center, std, data):
            knn = KNN(2, data)
            dist_between = knn.get_euclidean_distance(input, center)
          #  print(type(input[1]))
            #print(type(center[1]))
           # print(dist_between)
            output = np.exp(-1 / (2 * std ** 2) * dist_between ** 2)
            # print(output)
            return output


    ":param medoids_list,  a list of medoids in a cluster"
    # function to get max dist
    def getMaxDist(self, medoids_list, data):
        maxDist = 0
        knn = KNN(2, data)
        for medoid in medoids_list:
            for medoid2 in medoids_list:
                # compare against all other medoids
                curDist = knn.get_euclidean_distance(medoid, medoid2)
                if curDist > maxDist:
                    maxDist = curDist
       # print(maxDist)
        return maxDist

    def getMaxDistMeans(self, mean_list, data):
        maxDist = 0
        knn = KNN(2, data)
        for clust in mean_list:
            for clus2 in mean_list:
                # compare against all other medoids
                curDist = knn.get_euclidean_distance()
                if curDist > maxDist:
                    maxDist = curDist
        # print(maxDist)
        return maxDist

    """
    :param data_set : data set to train on
    :param expected_value : expected values for that data set
    :param data_instance : instance of data class
    """
    def trainReg(self, data_set, expected_values, data_instance):


        kmean = Kmeans(self.clusters, data_instance)
        medoids_list = kmean.k_means(data_set, self.clusters)
        print(medoids_list)
        #pam = PAM(k_val=self.clusters, data_instance=data_instance)
        #medoids_list, filler = pam.assign_random_medoids(data_set, self.clusters)
        #pam.assign_data_to_medoids(data_set, medoids_list)
        # set the STD of the clusters  (doing once for now)
        self.std = self.getMaxDist(medoids_list, data_instance) / np.sqrt(2 * self.clusters)
        self.medoids = medoids_list

        # train the regression model
        converged = False
        iterations = 0
        print("Training the RBF net")
        while not converged:

            if(iterations % 100 is 0):
                print("Iteration: ")
                print(iterations)
            for index, row in data_set.iterrows():
                # calculate the activation functions for each of the examples for each hidden node (cluster)
                a = []

                for medoid in medoids_list:
                    medoidAct = self.calcHiddenOutputs(row,medoid, self.std, data_instance)
                    # print(medoidAct)
                    a.append(medoidAct)


                a = np.array(a)
                #(a.T.dot(self.weights))

                F = a.T.dot(self.weights) + self.bias

                 #backwards update pass
                error = -(expected_values[index] - F)

                self.weights = self.weights - self.learning_rate * a * error
                self.bias = self.bias - self.learning_rate * error


            if iterations > self.maxruns:
                # break out if we hit the maximum runs
                converged = True

            iterations += 1


    # function to predict values given current weights and a dataset
    def predictReg(self, data_set, data_instance):
        predictions = []
        print("Calculating Predicted Values")
        for index, row in data_set.iterrows():
            a = []

            for medoid in self.medoids:
                medoidAct = self.calcHiddenOutputs(row, medoid, self.std, data_instance)

                a.append(medoidAct)

            a = np.array(a)
            F = a.T.dot(self.weights) + self.bias

            predictions.append(F.item())
        return predictions


    def mean_squared_error(self, predicted_data, actual_data):
        """
        :param predicted_data:  list of predicted values for datapoints (assume same order)
        :param actual_data: actual values for those said data points  (assume same order)
        :return MSE from the predicted data
         """
        print("Calculating Mean Squared Error")
        n = len(actual_data)  # get out n for MSE
        sum = 0  # sum of the MSE squared values

        for (predict, true) in zip(predicted_data, actual_data): # go through all the points at the same time
            currentSum = (true - predict) ** 2  # square it
            sum += currentSum # add current to total sum

        # divide by n
        sum = sum/n
        return sum # done, return sum