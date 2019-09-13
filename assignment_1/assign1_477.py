import pandas as pd
import math
import numpy as np

class data_proces():
    def __init__(self):
        self.my_soy = None
        self.my_glass = None
        self.my_votes = None
        self.my_cancer = None
        self.my_iris = None



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
        # self.my_iris = self.my_iris.replace(to_replace='Iris-setosa', value=1)
        # self.my_iris = self.my_iris.replace(to_replace='Iris-versicolor', value=2)
        # self.my_iris = self.my_iris.replace(to_replace='Iris-virginica', value=3)
        # do something with glass data

    def removeMissing(self):   # remove missing values, needs to be done for each array:  Methodology: generate average for attribute and use that
        # go through rows and check for NaN values, if nan change to average for that column (attribute)
        data_list = [self.my_votes, self.my_iris, self.my_cancer, self.my_soy,self.my_glass]  # list of dataframes to go through

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
    # end of class for doing data stuff (for now)


data = data_proces()
data.__init__()
data.loadData()
data.miscDataWork()
# order of data_list:  self.my_votes, self.my_iris, self.my_cancer, self.my_soy,self.my_glass
data_list = data.removeMissing()
# print(str(data_list[0]))
# order of data sets matches the main sets, 0 - 4 matches above
test_data, training_data = data.splitData(data_list)


class naive_bayes():  # implement naive bayes here

    def prepPredict(self, train_data): # split data into sets, done  as class names differ by set
        class_data_list = []  # list of data by class
        class_atribute_vals = {}  # number of examples in a class / training set example number
        for tdata in train_data:
            class_list = tdata[(len(tdata.columns)-1)].unique()
            class_attr_num = []  # middle ground var to hold examples/total for a given datas et

            for class_name in class_list:  # split data by class
                class_data_list.append(tdata.loc[tdata[len(tdata.columns)-1] == class_name])
                class_atribute_vals.update({class_name: ((len(tdata.loc[tdata[len(tdata.columns)-1] == class_name]))/(len(tdata)))})
                # use dictionary to store class values for later

        return class_atribute_vals, class_data_list

    def calcProbs(self, class_data):
        # calculate attribute value probabilites
        class_examples = len(class_data)  # total examples in class
        valueprobabilites = []  # list of dictionaries to hold probabilites
        for myClass in class_data:  # go through all class data and calc probs, store prob in dict
            ClassList = []
            for col in myClass.iloc[:, :-1]: # go through each col and get unique attr vals (ignore class col)
                attr_vals_list = myClass[col].unique()
                for val in attr_vals_list:  # go attr vals and divide number of mathing examples by total in class
                    matching_ex = myClass[myClass[col] == val].shape[0]
                    # calculate prob using what we figured out, using len(dataset)-1 to account for class col
                    attr_val_prob = (matching_ex + 1)/(class_examples +(len(myClass) - 1))
                    # store this val, with appropriate keywords, {str"Attr:AttrVal": prob}
                    attr_dict = {str(col) + ":" + str(val):attr_val_prob}
                    ClassList.append(attr_dict)

            valueprobabilites.append(ClassList)
        return valueprobabilites

    # now we need to actually make our predictions based on training set data

n = naive_bayes()
classAttrVals, class_data = n.prepPredict(training_data)
valProbs = n.calcProbs(class_data)


