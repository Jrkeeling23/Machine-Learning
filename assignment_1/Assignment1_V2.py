import operator

import pandas as pd
import math
import numpy as np


class NV:

    def loadData(self):  #assuming here that data is stored in the data/filename/file format
        self.my_soy = pd.read_csv('/home/alex/Documents/Montana State/CSCI_447_Machine_Learning/assignment_1/data/soybean-small.data', header=None)
        self.my_glass = pd.read_csv('/home/alex/Documents/Montana State/CSCI_447_Machine_Learning/assignment_1/data/glass.data', header=None)
        self.my_votes = pd.read_csv('/home/alex/Documents/Montana State/CSCI_447_Machine_Learning/assignment_1/data/house-votes-84.data', header=None)
        self.my_cancer = pd.read_csv('/home/alex/Documents/Montana State/CSCI_447_Machine_Learning/assignment_1/data/breast-cancer-wisconsin.data', header=None)
        self.my_iris = pd.read_csv('/home/alex/Documents/Montana State/CSCI_447_Machine_Learning/assignment_1/data/iris.data', delimiter=',', header=None)

    def miscDataWork(self):  # turn data into discrete format here (and any other odd work)
        # change to discrete values, need to change votes and glass data, make final row class row and numbered
        self.my_votes = self.my_votes.replace(to_replace='y', value=1)
        self.my_votes = self.my_votes.replace(to_replace='n', value=0)
        # setting '?' to 2 in votes as it is not a "missing" value but a third choice
        self.my_votes = self.my_votes.replace(to_replace='?', value=2)
        # should not need to do this as not calculating average will ignore last row
        # self.my_votes = self.my_votes.replace(to_replace='democrat', value=1)
        # self.my_votes = self.my_votes.replace(to_replace='republican', value=0)

        # change fist row to last row for format
        cols = self.my_votes.columns.tolist()  # get cols
        cols = cols[1:] + cols[:1]  # re order
        self.my_votes = self.my_votes[cols]  # assign new order to my_votes
        self.my_votes.columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]  # manually fixes broke cols
        # fix iris data (set last 3 columns to 1-3: NOTE: Note needed as we can ignore last col

    def removeMissing(
                self):  # remove missing values, needs to be done for each array:  Methodology: generate average for attribute and use that
            # go through rows and check for NaN values, if nan change to average for that column (attribute)
            # data_list = [self.my_votes, self.my_iris, self.my_cancer, self.my_soy,self.my_glass]  # list of dataframes to go through
            data_list = [self.my_votes, self.my_soy]
            for my_list in data_list:  # go through our datasets
                for row in range(my_list.shape[0]):  # go through rows
                    for col in range(my_list.shape[1]):  # go through cols
                        true_val = my_list.at[row, col]  # var for holding col val (str int compare issues)
                        if str(true_val) is '?':  # check if it is a "missing" value
                            col_av = 0  # init for column average
                            # get the average of the column
                            no_str = [x for x in my_list[col].tolist() if
                                      not isinstance(x, str)]  # get a list of all non str values (not '?')
                            if len(no_str) > 0:  # make sun length of that list is > 0 to avoid divide by 0
                                # NOTE::  Only time we should see a length 0 with no nums would be in the class rows
                                # so for now ignoring the "else" case for this if as irelevent
                                col_av = (sum(no_str) / len(
                                    no_str)).__round__()  # get the average of that list and round it

                            my_list.at[row, col] = col_av  # set our new row to the new average

            # set class data list equal to data_list from method
            return data_list

    def splitData(self, data_list):
            # split into test and training data
            test_data = []
            training_data = []
            # use numpys split with pandas sample to randomly split the data
            for my_list in data_list:  # split each list into 2 lists, training and test and make a list for each
                training_data_temp, test_data_temp = np.split(my_list.sample(frac=1), [int(.8 * len(my_list))])
                training_data.append(training_data_temp)  # append our test sets and training sets to their lists
                test_data.append(test_data_temp)

            return test_data, training_data
        # end of  doing data stuff (for now)




        # start of algorithm things
    # ONE DATASET AT A TIME
    def calcExamples(self, dataset):
        class_list = dataset[(len(dataset.columns) - 1)].unique()
        class_atribute_vals = {}  # number of examples in a class / training set example number
        for class_name in class_list:  # split data by class
          #  class_atribute_vals.update({class_name: ((len(dataset.loc[dataset[len(dataset.columns) - 1] == class_name])) / (len(dataset)))})

            mysum = dataset.loc[dataset.iloc[:, -1] == class_name].count().sum()

            examples = dataset.count().sum()


            classVal = {class_name: mysum/ examples}

            class_atribute_vals.update(classVal)
        return class_atribute_vals



    def seperateByClass(self , dataset):
        class_list = dataset[(len(dataset.columns) - 1)].unique()
        DataSet = []
        for class_name in class_list:
            DataSet.append(dataset.loc[dataset[len(dataset.columns) - 1] == class_name])

        return DataSet

    def calcProbs(self, sep_dataset):  # calculate probabilites  given class seperated dataset
        probs = {}
        for class_set in sep_dataset:
            # for each attribute value. divide number of examples that that attribute value plus one by the number
            # of exammples in the class
            classArr = class_set[(len(class_set.columns) - 1)].unique()
            ClassName = classArr[0]
            #print(ClassName)
            # go through cols (attributes) but not class col
            for column in class_set.iloc[:, :-1]:
              values = class_set[column].unique() # find unique values
              for val in values:  # find out how many times that value appears in this col
                valcount = (class_set[column] == val).sum()
                # divide that count there by the count we get for all values in the class (in our case just all vals)
                total_examples = (class_set.count()).sum()
                attribute_count = len(class_set.index)  # counting rows
                # do the division
                value_probability = (valcount + 1) / (total_examples + attribute_count)
                # store the class, value (what it was), colum (what attr it is) and prob
                probs[ClassName, val, column] = value_probability

            # prob_dict = {}  # dictionary to store probs
             #for prob_data in sep_dataset:
                #   for useless, row in prob_data.iterrows():  # go through rows
           #     for i in range(0, len(prob_data.columns)-2): # go through cols, ignore class col
          #               curVal = row[i]

            #        attr_class = row[len(prob_data.columns)-1]
                    # calc number attributes matching current
             #       matching_ex = prob_data[prob_data == curVal].count()
                #    matching_ex = matching_ex.sum()
              #      # calc number of attributes in class
               #     total_attr = prob_data.iloc[:, :-1].count().sum()
                    #print(total_attr)# all but final row
                 #   attr_prob = (matching_ex + 1)/(total_attr + len(prob_data.columns)-1)
                    # now to store attr prob + class
                    # class, index, value : prob
                  #  prob_dict[attr_class, i, curVal] = attr_prob
                    #print(attr_prob)

        return probs

    def predictData(self, test_set, attr_probs, class_attrVal):

        class_names = (test_set[len(test_set.columns) - 1]).unique()
        predicSet = []

        # go through the row and predict stuff
        for curObs, row in test_set.iterrows():
            classDict = {}
            finalPred = {}
            for curClass in class_names:

                curProb = 0
                curClassProbs = []

                for i in range(0, (len(test_set.columns) -2)):
                    indexVal = row[i]
                    index = i;
                    # get all stored values that have same classname, index and value and add probability to list
                    try:
                        curProb = attr_probs[curClass, indexVal, index]
                    except:
                        print("Value not found in probabilites")

                  #  print(curProb)
                    # add curProb to this classes list of prob
                    # do not need index as we check this for other classes but not anymore then that
                    curClassProbs.append(curProb)


                classDict[curClass] = curClassProbs  # set dict probabilites for current class equal to class prob list

            # we got the probabilites for that val, now we need to find out what class has highest prob for current obs
            for probClass in class_names:
                # take all values in one set and multiply them
                prediction = class_attrVal[probClass] * (np.prod(classDict[probClass])) # sum all vals
                finalPred[prediction] = {probClass}



        # last step, choose larger probability
            maxProb = max(k for k, v in finalPred.items() if v != 0)
            maxProbClass = finalPred[maxProb]

            predicSet.append([maxProb, curObs, maxProbClass])
            print("Final pred: ", finalPred)
        return predicSet

    def accuracy(self, expected, predicted): # Function to get the accuracy of the algorithm

        predict = []
        for i in predictions:  # grabs the prediction value from the data set
            predict.append(i[2])  # Appends the predictions to a list

        # Counters for the accuracy formula
        true_positive = 0
        false_positive = 0

        for i in range(len(predict)):  # Loops through the predictions/expected values and matches a true or false.
            if expected[i] in predict[i]:  # checks if the exptected output matches the prediction
                true_positive += 1
            else:
                false_positive += 1

        accuracy = true_positive / (true_positive + false_positive) # Calculates the accuracy
        print("Accuracy: ", accuracy)

# data stuff
n = NV()
n.loadData()
n.miscDataWork()
data_list = n.removeMissing()
test_data, training_data = n.splitData(data_list)



# actual algo stuff  (just 1 dataset to test)
attr_eg = n.calcExamples(training_data[1])
print("attr", attr_eg)
class_data = n.seperateByClass(training_data[1])

# print(attr_eg)
probsList = n.calcProbs(class_data)

expected = np.array(test_data[1][len(test_data[1].columns) - 1]) # the list of expected outputs from the test data
predictions = n.predictData(test_data[1], probsList, attr_eg) # list of predicted outputs

n.accuracy(expected, predictions)



