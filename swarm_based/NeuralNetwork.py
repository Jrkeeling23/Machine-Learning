import random
import math
import numpy as np


class NeuralNetwork:
    def __init__(self, data_instance): #, edited_data, compressed_data, centroids_cluster, medoids_cluster):
        self.data_instance = data_instance
        self.layers = []
        self.personal_best = None
        self.output_vector = None
        self.fitness = None
        self.velocity = None
        # self.edited_data = edited_data
        # self.compressed_data = compressed_data
        # self.centroids_cluster = centroids_cluster
        # self.medoids_cluster = medoids_cluster

    # self.layers = []

    def make_layers(self, no_of_layers, no_of_nodes):
        """
        :param no_of_layers: sets up the number of hidden layers for the network
        :param no_of_nodes: sets up the number of nodes in the hidden layer
        :return:
        """
        # row = self.data_instance.df.shape[0]-1
        first_layer_size = self.data_instance.df.shape[1]-1
        self.layers = []
        self.layers.append(Layer(first_layer_size))
        self.layers[0].make_nodes()
        if no_of_layers!=0:
            for i in range(no_of_layers):
                self.layers.append(Layer(no_of_nodes))
                self.layers[i+1].make_nodes()
                if i+1 == no_of_layers:
                    self.output_vector = self.set_output()
                    self.layers.append(Layer(len(self.output_vector)))
                    self.layers[i+2].make_nodes()
                for j in range(len(self.layers[i].nodes)):
                    for f in range(len(self.layers[i+1].nodes)):
                        self.layers[i].nodes[j].outgoing_weights.append(Weight(self.layers[i].nodes[j], self.layers[i+1].nodes[f]))
                for f in range(len(self.layers[i+1].nodes)):
                    for j in range(len(self.layers[i].nodes)):
                        self.layers[i+1].nodes[f].incoming_weights.append(self.layers[i].nodes[j].outgoing_weights[f])
        else:
            self.output_vector = self.set_output()
            self.layers.append(Layer(len(self.output_vector)))
            self.layers[1].make_nodes()
            for j in range(len(self.layers[0].nodes)):
                for f in range(len(self.layers[1].nodes)):
                    self.layers[0].nodes[j].outgoing_weights.append(Weight(self.layers[0].nodes[j], self.layers[1].nodes[f]))
            for f in range(len(self.layers[1].nodes)):
                for j in range(len(self.layers[0].nodes)):
                    self.layers[1].nodes[f].incoming_weights.append(self.layers[0].nodes[j].outgoing_weights[f])
        return self.layers, self.output_vector

    def vectorize(self):  # , layers):
        vector = []
        for i in range(1, len(self.layers), 1):
            for n in range(len(self.layers[i].nodes)):
                for w in range(len(self.layers[i].nodes[n].incoming_weights)):
                    vector.append(self.layers[i].nodes[n].incoming_weights[w].weight)
                vector.append(self.layers[i].nodes[n].bias)
        return vector

    def networkize(self, vector):  # layers, vector):
        j=0
        new_network = self.layers.copy()
        for i in range(1, len(new_network), 1):
            for n in range(len(new_network[i].nodes)):
                for w in range(len(new_network[i].nodes[n].incoming_weights)):
                    new_network[i].nodes[n].incoming_weights[w].weight = vector[j]
                    j+=1
                new_network[i].nodes[n].bias = vector[j]
                j+=1
        return new_network

    def GADEnet(self, layers, vector):
        j = 0
        new_network = layers.copy()
        for i in range(1, len(new_network), 1):
            for n in range(len(new_network[i].nodes)):
                for w in range(len(new_network[i].nodes[n].incoming_weights)):
                    new_network[i].nodes[n].incoming_weights[w].weight = vector[j]
                    j += 1
                new_network[i].nodes[n].bias = vector[j]
                j += 1
        return new_network

    def GADEvec(self, layers):
        vector = []
        for i in range(1, len(layers), 1):
            for n in range(len(layers[i].nodes)):
                for w in range(len(layers[i].nodes[n].incoming_weights)):
                    vector.append(layers[i].nodes[n].incoming_weights[w].weight)
                vector.append(layers[i].nodes[n].bias)
        return vector

    def sigmoid(self, input):
        # i = 0
        self.layers[0].make_input_layer(input)
        for layer in self.layers[1:]:
            for node in layer.nodes:
                sigmoid_total = 0
                for weight in node.incoming_weights:
                    sigmoid_total += weight.get_weight() * weight.L_neuron.value
                sigmoid_total += node.bias
                node.value = 1/(1 + np.exp(-sigmoid_total))

        output = []
        for node in self.layers[-1].nodes:
            output.append(node.value)
        return output

    def prediction(self, outputs, output_values):
        guess = 0
        for i in range(len(outputs)):
            if output_values[i] > guess:
                guess = i
        return outputs[guess]

    def cost(self, output_values, outputs, expected):
        high_value = 0
        for i in range(len(outputs)):
            if outputs[i] == expected:
                high_value = i
        compare = []
        for j in range(len(output_values)):
            if j != high_value:
                compare.append(float(0))
            else:
                compare.append(float(1))
        cost = 0
        for f in range(len(output_values)):
            cost += (output_values[f]-compare[f]) ** 2
        return cost, compare



    def set_output(self):
        output = []
        label = self.data_instance.label_col
        if not self.data_instance.regression:
            for index, row in self.data_instance.train_df.iterrows():
                if row[label] not in output:
                    output.append(row[label])
        else:
            for index, row in self.data_instance.train_df.iterrows():
                if row[label] not in output:
                    output.append(row[label])
        return sorted(output)

    def gradient_descent(self, layers, eta, alpha, comparison, group):
        for f in range(len(group)):
            self.backpropagation(layers, group[f], comparison[f])
        for i in range(1, len(layers)-1, 1):
            for node in layers[i].nodes:
                # node.bias +=
                node.prev_bias_change = (-eta*node.bias_change/len(group)+alpha*node.prev_bias_change)
                node.bias += node.prev_bias_change
                node.bias_change = 0
                for weight in node.incoming_weights:
                    weight.prev_change = (-eta*weight.weight_change/len(group)+alpha*weight.prev_change)
                    weight.weight += weight.prev_change
                    weight.weight_change = 0
        return  # need to get something to return still

    def backpropagation(self, layers, input, compare):
        self.sigmoid(input)
        for i in range(len(layers)-1, 1, -1):
            if i == len(layers)-1:
                j=0
                for node in layers[i].nodes:
                    node.delta = -(compare[j] - node.value) * node.value * (1 - node.value)
                    node.bias_change += node.delta
                    j += 1
                    for weight in node.incoming_weights:
                        weight.weight_change += node.delta * weight.weight
            else:
                for node in layers[i].nodes:
                    summer = 0
                    for weight in node.outgoing_weights:
                        summer += weight.R_neuron.delta * weight.weight
                    node.delta = node.value * (1 - node.value) * summer
                    node.bias_change += node.delta
                    for weight in node.incoming_weights:
                        weight.weight_change += node.bias * weight.L_neuron.value
        return



class NetworkClient:
    def __init__(self, data_instance):
        self.data_instance = data_instance

    def prepare(self):
        self.data_instance.split_data()
        if self.data_instance.regression:
            self.data_instance.regression_data_bins(9, True)

    def train_it(self, hidden_layers, hidden_nodes, eta, alpha, stoch):
        saved = None
        network = NeuralNetwork(self.data_instance)
        layers, output_layer = network.make_layers(hidden_layers, hidden_nodes)
        output_predictions = []
        costs = []
        compare = []
        for index, row in self.data_instance.train_df.iterrows():
            output_predictions.append(network.sigmoid(row.drop(self.data_instance.label_col)))
            cos, comp = network.cost(output_predictions[-1], output_layer, row[self.data_instance.label_col])
            costs.append(cos)
            compare.append(comp)
        tries = 0
        while True:
            tries += 1
            # print(tries)
            # f=0
            group = []
            comp_group = []
            check_group = []
            j = 0
            for index, row in self.data_instance.train_df.iterrows():
                j += 1
                if j % stoch == 0:
                    #changes, length = network.gradient_descent(layers, eta, alpha, comp_group, group)
                    network.gradient_descent(layers, eta, alpha, comp_group, group)
                    # print(check_group)
                    # print(output_layer)
                    # print(comp_group)
                    group = []
                    comp_group = []
                group.append(row.drop(self.data_instance.label_col))
                check_group.append(row[self.data_instance.label_col])
                cos, comp = network.cost(output_predictions[-1], output_layer, row[self.data_instance.label_col])
                comp_group.append(comp)
                # for j in range(stoch):
                #     group.append(costs[f+j])
                #     comp_group.append(compare[f+j])
                # f+=1
            # for i in range(0, len(costs), stoch):
            #     group = []
            #     comp_group = []
            #     for j in range(stoch):
            #         if i+j < len(costs):
            #             # print(len(costs))
            #             # print(i+j)
            #             group.append(costs[i+j])
            #             comp_group.append(compare[i+j])
            #
            #     changes, length = network.gradient_descent(layers, group, eta, alpha, comp_group)
            #     # print(changes)
            output_predictions = []
            costs = []
            for index, row in self.data_instance.train_df.iterrows():
                output_predictions.append(network.sigmoid(row.drop(self.data_instance.label_col)))
                costs.append(network.cost(output_predictions[-1], output_layer, row[self.data_instance.label_col])[0])
            # print(changes)
            if saved is None:
                saved = costs
            else:
                if tries % 100 == 0:
                    print("Summed Saved: ", sum(saved), " Summed Costs: ", sum(costs))
                if sum(saved) > sum(costs):
                    saved = costs

                else:
                    break

            checker = .0005
            test = 0
            # for i in range(len(changes)):
            #     my_break = False
            #     test += abs(changes[i])
            #  TODO: put in if statement checking cost
            #if all(abs(x) <= checker for x in changes):  # changes.all() <= checker:  # abs(changes[i])
                # my_break = True
               # break
        # print(my_break)
        # if my_break:
        #     break
        return layers, output_layer, network

    def testing(self, layers, output_set, network):
        correct = 0
        total = 0
        for index, row in self.data_instance.test_df.iterrows():
            output_prediction = network.sigmoid(row.drop(self.data_instance.label_col))
            if network.prediction(output_set, output_prediction) == row[self.data_instance.label_col]:
                correct += 1
            total += 1
        # print( "Correctly Predicted ", correct, " Total Predicted ", total)
        return (correct/total)


class Layer:
    def __init__(self, no_of_nodes):
        self.no_of_nodes = no_of_nodes
        self.nodes = []

    def make_nodes(self):
        for nodes in range(self.no_of_nodes):
            self.nodes.append(Neuron(float(random.randint(-1, 1))/100))

    def make_input_layer(self, inputs):
        i = 0
        for input in inputs:
            self.nodes[i].value = input
            i += 1


class Neuron:
    def __init__(self, bias, value=None):
        self.bias = bias
        self.prev_bias_change = 0
        self.bias_change = 0
        self.is_sigmoidal = None
        self.is_linear = None
        self.incoming_weights = []
        self.outgoing_weights = []
        self.value = value
        self.delta = 0


class Weight:
    def __init__(self, L_neuron, R_neuron):
        self.L_neuron = L_neuron
        self.R_neuron = R_neuron
        self.weight = random.uniform(-1, 1)
        self.weight_change = 0
        self.prev_change = 0
        self.momentum_cof = .5
        self.eta = .1

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight
