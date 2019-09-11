import pandas as pd
import math

class data_proces():
    def __init__(self):
        self.my_soy = None
        self.my_glass = None
        self.my_votes = None
        self.my_cancer = None
        self.my_iris = None



    def loadData(self):  #assuming here that data is stored in the data/filename/file format
        self.my_soy = pd.read_csv('data/soybean-small.data', header=None)
        self.my_glass = pd.read_csv('data/glass.data', header=None)
        self.my_votes = pd.read_csv('data/house-votes-84.data', header=None)
        self.my_cancer = pd.read_csv('data/breast-cancer-wisconsin.data', header=None)
        self.my_iris = pd.read_csv('data/iris.data', delimiter=',', header=None)

    def miscDataWork(self):  # turn data into discrete format here (and any other odd work)
        # change to discrete values, need to change votes and glass data, make final row class row and numbered
        self.my_votes = self.my_votes.replace(to_replace='y', value=1)
        self.my_votes = self.my_votes.replace(to_replace='n', value=0)
        self.my_votes = self.my_votes.replace(to_replace='democrat', value=1)
        self.my_votes = self.my_votes.replace(to_replace='republican', value=0)
        # change fist row to last row for format
        cols = self.my_votes.columns.tolist()  # get cols
        cols = cols[1:] + cols[:1]  # re order
        self.my_votes = self.my_votes[cols]  # assign new order to my_votes
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
        for my_list in data_list:  # split each list into 2 lists, training and test and make a list for each
            # rows = my_list.shape[0]  # get rows for total amount of data
            splitnum = math.ceil(len(my_list.index) * .8)  # create a index that contains 80% of the data
            print(splitnum)
            training_data_temp = my_list[:splitnum]
            test_data_temp = my_list[splitnum:]

            training_data.append(training_data_temp)  # append our test sets and training sets to their lists
            test_data.append(test_data_temp)

        return test_data, training_data
    # end of class for doing data stuff (for now)


data =  data_proces()
data.__init__()
data.loadData()
data.miscDataWork()
data_list = data.removeMissing()
test_data, training_data = data.splitData(data_list)

print(training_data)

class naive_bayes:  #implement naive bayes here
    def __init__(self):
        self.placeholder = None

