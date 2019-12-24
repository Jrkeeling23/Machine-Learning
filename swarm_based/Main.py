from Data import Data
import pandas as pd
import csv
from DE import  DE
from GA import GA
from NeuralNetwork import NeuralNetwork
from NeuralNetwork import  NetworkClient

def load_data():
    """
    loads the data (csv) files
    :return: list of Data instances
    """
    with open('data/segmentation.data') as fd:
        reader = csv.reader(fd)
    data_list = [Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False),
                 Data('car', pd.read_csv(r'data/car.data', header=None), 5, False),
                 Data('segmentation', pd.read_csv(r'data/segmentation.data', header=None, skiprows=4), 0, False),
                 Data('machine', pd.read_csv(r'data/machine.data', header=None), 0, True),
                 Data('forest_fires', pd.read_csv(r'data/forestfires.data', header=None), 12, True),
                 Data('wine', pd.read_csv(r'data/wine.data', header=None), 0, True)]
    return data_list
    # cat: abalone, car, segmentation
    # reg: machine, forestfires, wine


def run_pop_algos_for_vid():
    "run the pop algos for proj4 vid"
    # setup data to use
    data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False)
    # take a sample as results do not matter for this
    df = data.df.sample(100)
    data.split_data(data_frame=df)
    gen_algo = GA(1000, 4, data, max_runs=1000, mutation_rate=1)
    print("----------------------- RUNNING THE GA -----------------")
    # get chromosome object from GA
    bestC = gen_algo.run_GA()
    print("Best fitting vector From the GA")
    print(bestC.net_vector)
    client = NetworkClient(data)
    network = NeuralNetwork(data)
    new_Net = network.GADEnet(bestC.layers, bestC.net_vector)
    print("Printing testing results from the GA")
    print(client.testing(new_Net, bestC.outputs, bestC.network))
    print("----------------------- GA DONE -----------------")
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print("----------------------- RUNNING DE -----------------")
    de_algo = DE(10, .7, 2, 4, data, max_runs=100, mutation_rate=.03)
    bestC = de_algo.run_DE()
    print("Best fitting vector from DE")
    print(bestC.net_vector)
    client = NetworkClient(data)
    network = NeuralNetwork(data)
    new_Net = network.GADEnet(bestC.layers, bestC.net_vector)
    print("Printing testing results from DE")
    print(client.testing(new_Net, bestC.outputs, bestC.network))
    print("----------------------- DE DONE -----------------")



class Main:
    def __init__(self):
        self.data_list = load_data()


    # def perform_KNN(self, k_val, query_point, train_data):

if __name__ == '__main__':
    run_pop_algos_for_vid()