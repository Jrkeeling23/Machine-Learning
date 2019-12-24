from NeuralNetwork import NeuralNetwork
import random

class PSO:
    def __init__(self, data_instance, no_of_particles, no_of_nodes, no_of_layers):
        self.data_instance = data_instance
        self.particles = []
        for i in range(no_of_particles):  # initializing all the particles
            self.particles.append(NeuralNetwork(self.data_instance))
            self.particles[i].make_layers(no_of_layers, no_of_nodes)
        self.group_best = None

    def move_them(self, c1, c2, omega):
        print(self.particles[0].fitness)
        for particle in self.particles:  # going through the particles
            fit = 0
            dumb = 0
            for index, row in self.data_instance.train_df.iterrows():
                acc = self.accuracy(particle.sigmoid(row.drop(self.data_instance.label_col)), particle.output_vector, row[self.data_instance.label_col])
                fit += acc
                dumb += 1
            particle.fitness = fit/dumb  # finding the fitness for each particle

            if particle.personal_best == None:
                particle.personal_best = [particle.vectorize(), particle.fitness]  # updating pBest if necessary
            elif particle.personal_best[1]<particle.fitness:
                particle.personal_best = [particle.vectorize(), particle.fitness]

            if self.group_best == None:
                self.group_best = [particle.vectorize(), particle.fitness]  # updating gBest if necessary
            elif self.group_best[1]<particle.fitness:
                self.group_best = [particle.vectorize(), particle.fitness]

        for particle in self.particles:  # going through each particle to update velocity and position
            r1 = random.uniform(0, 1)  # stochastic variables
            r2 = random.uniform(0, 1)
            vector = particle.vectorize()  # converting to vector
            var1 = c1*r1
            var2 = c2*r2
            print("Location before movement:", vector)
            print("Cognitive component and stochastic multiplier:", c1, r1)
            print("Social component and stochastic multiplier:", c2, r2)
            if particle.velocity == None:  # initializes velocities to 0
                particle.velocity = [0]*len(vector)
            print("Previous velocity:", particle.velocity)
            for loc in range(len(vector)):  # updating velocity vector according to equation
                particle.velocity[loc] = omega*particle.velocity[loc]+var1*(particle.personal_best[0][loc]-vector[loc])+var2*(self.group_best[0][loc]-vector[loc])
            print("Current velocity:", particle.velocity)
            for i in range(len(vector)):  # updating position vector
                vector[i] = vector[i]+particle.velocity[i]
            particle.layers = particle.networkize(vector)  # converting vector back to network
            print("Location after movement:", vector)
        print(self.particles[0].fitness)

    def accuracy(self, pred_output, output_layer, actual):  # function for going through the input to determine if it is correct or not
        high_value = 0
        for i in range(len(output_layer)):
            if output_layer[i] == actual:
                high_value = i
        prediction = 0
        for f in range(len(pred_output)):
            if pred_output[f] > pred_output[prediction]:
                prediction = f
        if prediction != high_value:
            return 0
        else:
            return 1
