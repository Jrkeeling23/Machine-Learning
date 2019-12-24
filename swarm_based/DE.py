import pandas as pd
import numpy as np
import random
import math
from NeuralNetwork import  NeuralNetwork
from NeuralNetwork import  NetworkClient
from Data import Data

# class to represent chromosomes
class Chromosome:
    # net_vector represents the veector that this chromosome holds
    # layers represents the layers the NN contains, used to turn back into a NN form
    # network the NN this holds, used to turn into vector or into layer form
    # fitness represents the fitness value of that vector
    # outputs is used to calculate the fitness of the chromo
    def __init__(self, net_vector, network, layers, outputs, fitness = 0,):

        self.net_vector = net_vector
        self.network = network
        self.fitness = fitness
        self.layers = layers
        self.outputs = outputs

class DE:
    # t_size is the size of a tournement
    # cross over prob is the probability used to select a gene from parent 1
    # mutation rate is the chance of mutation
    # pop size is size of population
    # population is list of chromosomes
    # data is a data instance
    # layers is # of layers for NN's
    # nodes is # of nodes in NN's
    def __init__(self,pop_size, beta, trial_vectors, t_size, data, layers=2, nodes=5, crossover_prob=.5, mutation_rate=.01, max_runs=1000):
        self.t_size = t_size
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.pop_size = pop_size
        self.population = []
        self.data = data
        self.layers = layers
        self.nodes = nodes
        self.max_runs = max_runs
        self.beta = beta
        self.trial_vectors = trial_vectors

    # calculate fitness given some target chromosome
    def CalcFitness(self, network, layers, outputs):
        client = NetworkClient(self.data)
        fitness = client.testing(layers, outputs, network)
        # print("Fitness for this individual is: ")
        # print(fitness)
        print("Calculated Fitness ", fitness)
        return fitness

    # init the population of the GA
    def init_pop(self):
        data = self.data
        # counter for below loop
        counter = 0
        print("Creating Population")
        while counter < self.pop_size:
            # create a new random network, create a new chromosome with it
            network = NeuralNetwork(data_instance=data)
            layers, outputs = network.make_layers(self.layers, self.nodes)
            net_vector = network.GADEvec(layers)
            newPop = Chromosome(net_vector, network, layers, outputs)
            self.population.append(newPop)
            # calculate fitness of each pop member
            newPop.fitness = self.CalcFitness(network, layers, outputs)


            counter+=1
        print("Done Creating Population")


    def Tourny_Selection(self):
        tourny = []
        indexs = []
        print("Performing Tournament Selection")
        #print(self.population)
        for i in range(self.t_size):  # tourny to select first parent
            # randomly generate index and add it to tourny
            # add the population member at that random num to tourny1 list
            trandom = random.randint(0, len(self.population)-1)
            #print(trandom)
            randomChrome = self.population[trandom]
            #print(randomChrome)
            #print(type(randomChrome))
            #print("Actual pop member")
           # print(self.population[trandom])
            #print(type(self.population[trandom]))

            tourny.append(randomChrome)
            indexs.append(trandom)



        # get the best of each tourny and use them as parents
        # set first one as most fit so far
        #print(tourny)
        bestSeen = tourny[0]
        bestIndex = indexs[0]
        for chrome in tourny:
            if chrome.fitness > bestSeen.fitness:
                bestSeen = chrome
                bestIndex = tourny.index(chrome)

        print ("Returning most fit example in tournament ")
        # return best value
        return bestSeen, bestIndex

    def Binomial_crossover(self, parent1, parent2):
        # new empty lists for the vectors of the children
        # only want 1 child for DE
        childVector1 = []
        # perform binomial crossover
        print("Performing Binomial Crossover")
        # print(parent1)
        # print(parent2)
        for num1, num2, in zip(parent1.net_vector, parent2.net_vector):
            rand = random.uniform(0, 1)
            if rand <= self.crossover_prob:
                childVector1.append(num1)
            else:
                childVector1.append(num2)

        # perform mutation
        print("Performing Mutation on the Children")
        for i in range(len(childVector1)):
            rand = random.uniform(0, 1)
            if rand <= self.mutation_rate :
                # currently limiting the size of the random to current val +- 2
                new_val = random.uniform(childVector1[i] - 2, childVector1[i] + 2)
                childVector1[i] = new_val



        # create two new chromosomes for each child
        child1 = Chromosome(childVector1, parent1.network, parent1.layers, parent1.outputs)
        # turn vectors into layers for fitness
        child1Layers = child1.network.GADEnet(child1.layers, child1.net_vector)
        # finish setting up fitness of each child
        child1.fitness = self.CalcFitness(child1.network, child1Layers, child1.outputs)
        # compare the fitness of children vs parents, replace parents if better
        ret_vector= None
        if child1.fitness >= parent1.fitness:
            # replace the parent if child better than current pop member (parent 1)
            ret_vector = child1
        else:
            ret_vector = parent1

        print("Finished Performing Crossover + Mutation")
        return ret_vector





    # run the DE Training algorithm
    def run_DE(self):
        self.init_pop()

        # counter for maxruns
        runs = 0
        while runs < self.max_runs:
            if runs % 100 == 0:
                print("---------------- Iteration: ", runs, " -----------------------------")
            runs += 1

            # go through pop and createe trial vectors

            for i in range(len(self.population)):

                # inital vector v sub 1
                vector1, index1 = self.Tourny_Selection()
                trial_vector = vector1.net_vector
                print("Creating Trial Vector")
                for i in range(self.trial_vectors):
                    # create a new trail vector and add to list (thye beta(x2 - x3) part
                    vector2, index2 = self.Tourny_Selection()
                    vector3, index3 = self.Tourny_Selection()

                    trail_vec_partial = self.beta * (np.subtract(vector2.net_vector, vector3.net_vector))

                    # now calculate the trial vector using these partial vectors
                    trial_vector = np.add(trial_vector, trail_vec_partial)

                # create a chromosome to represent trial vector
                trial_chromosome = Chromosome(trial_vector, self.population[i].network,
                                              self.population[i].layers, self.population[i].outputs)
                # get the network form of the vector
                trial_layers = trial_chromosome.network.GADEnet(trial_chromosome.layers, trial_chromosome.net_vector)
                # calculate fitness of that network
                trial_chromosome.fitness = self.CalcFitness(trial_chromosome.network, trial_layers,
                                                            trial_chromosome.outputs)
                # now we need to use mutation to generate a child vector using trail vector and current pops vector
                self.population[i] = self.Binomial_crossover(self.population[i], trial_chromosome)
                # moveing on to next pop memeber
                print("Current Population individual updated, moving on to next individual")

        # finished checking everything, return individual with highest fitness
        # go through population and find chromosome with highest fitness, return it's networkized form
        # randomly choose one to start
        bestChrome = self.population[random.randint(0, len(self.population)-1)]
        # go through all of the pop and return value with highest fitness
        for chrome in self.population:
            if chrome.fitness > bestChrome.fitness:
                print(chrome.fitness)
                bestChrome = chrome

        print("Returning Chromosome with highest fitness")
        # networkC = NeuralNetwork(self.data)
        # return networkC.networkize(bestChrome.layers, bestChrome.net_vector)
        return bestChrome