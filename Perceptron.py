from encodings.punycode import selective_find

__author__ = 'Ajinkya'

import math
import random
import collections
import copy


class Perceptron:

    output = 0
    connections = None

    def __init__(self, create):
        self.connections = collections.OrderedDict()
        if create == 10:
            self.connections['output_neuron_1'] = -0.53
            self.connections['output_neuron_2'] = -1.33
            self.connections['output_neuron_3'] = 1.10

        elif create == 11:
            self.connections['output_neuron_1'] = -1.38
            self.connections['output_neuron_2'] = -1.00
            self.connections['output_neuron_3'] = 1.20

        elif create == 12:
            self.connections['output_neuron_1'] = -1.22
            self.connections['output_neuron_2'] = -1.23
            self.connections['output_neuron_3'] = 1.19

        elif create == 13:
            self.connections['output_neuron_1'] = -1.31
            self.connections['output_neuron_2'] = -0.03
            self.connections['output_neuron_3'] = 0.42


        elif create == 2:

            self.connections['bias_hidden_neuron_1'] = 0.74
            self.connections['bias_hidden_neuron_2'] = 0.84
            self.connections['bias_hidden_neuron_3'] = 0.32
            self.connections['bias_hidden_neuron_4'] = 0.02

        elif create == 31:
            self.connections['hidden_neuron_1'] = 0.62
            self.connections['hidden_neuron_2'] = 0.83
            self.connections['hidden_neuron_3'] = 0.40
            self.connections['hidden_neuron_4'] = 0.33

        elif create == 32:
            self.connections['hidden_neuron_1'] = 0.14
            self.connections['hidden_neuron_2'] = 0.60
            self.connections['hidden_neuron_3'] = 0.94
            self.connections['hidden_neuron_4'] = 0.27

        elif create == 33:
            self.connections['hidden_neuron_1'] = 0.91
            self.connections['hidden_neuron_2'] = 0.28
            self.connections['hidden_neuron_3'] = 0.60
            self.connections['hidden_neuron_4'] = -0.23

        elif create == 5:
            self.connections['bias_output_neuron_1'] = -1.202
            self.connections['bias_output_neuron_2'] = -1.207
            self.connections['bias_output_neuron_3'] = -1.107



    #def
    def sigmoid(self, summation_input):
        '''
        This function calculates the sigmoid function for the give argument
        It also sets the output field of the Perceptron object
        :param summation_input:
        :return:
        '''
        temp = 1/(1 + math.pow(math.e, -summation_input))
        #print("****", self.output)
        return temp


class NN:

    input_layer = []            # stores the input neurons
    hidden_layer = []           # stores the hidden layer neurons
    output_layer = []           # stores the output layer neurons
    bias_for_hidden_layer = None    # object of Perceptron for bias of hidden layer
    bias_for_output_layer = None    # object of perceptron for bias of output layer

    input_neuron_1 = None
    input_neuron_2 = None
    input_neuron_3 = None
    temp_list = []

    list_of_inputs = []

    ## Constructor initializing the neurons
    def __init__(self, number_of_neurons):
        self.bias_for_hidden_layer = Perceptron(2)
        self.bias_for_output_layer = Perceptron(5)

        ## Creating input layer neuron
        self.input_neuron_1 = Perceptron(31)
        self.input_neuron_2 = Perceptron(32)
        self.input_neuron_3 = Perceptron(33)

        self.input_layer = [self.input_neuron_1, self.input_neuron_2, self.input_neuron_3]

        ## Creating hidden layer neuron
        self.hidden_layer.append(Perceptron(10))
        self.hidden_layer.append(Perceptron(11))
        self.hidden_layer.append(Perceptron(12))
        self.hidden_layer.append(Perceptron(13))

        ##Creating output layer neuron
        self.output_layer.append(Perceptron(1))
        self.output_layer.append(Perceptron(1))
        self.output_layer.append(Perceptron(1))

    def feed_forward(self, input1, input2, input3):
        '''
        This function implements the feed forward procedure of neural network

        :param input1: input 1 value for the neural network
        :param input2: input 2 value for the neural network
        :param input3: input 3 value for the neural network
        :return:    None
        '''

        global hidden_layer
        global output_layer

        counter = 0

        NN.list_of_inputs = [input1, input2, input3]

        ### Calculation for output of input to hidden layer
        for i in range(0, 4):
            if counter < 4:
                summation = self.input_neuron_1.connections.get("hidden_neuron_"+(str(counter+1))) * input1\
                            + self.input_neuron_2.connections.get("hidden_neuron_"+(str(counter+1))) * input2 + self.input_neuron_3.connections.get("hidden_neuron_"+(str(counter+1))) * input3

                summation += self.bias_for_hidden_layer.connections.get("bias_hidden_neuron_"+(str(counter+1))) * 1
                ii = self.hidden_layer[i].sigmoid(summation)
                self.hidden_layer[i].output = ii
                counter += 1

        counter1 = 0
        another = 0

        ### Calculation of output of hidden to output layer

        for i in range(0, 3):

            if counter1 < 3:
                total = self.hidden_layer[another].connections.get("output_neuron_"+str(counter1+1)) * self.hidden_layer[another].output + \
                            self.hidden_layer[another+1].connections.get("output_neuron_"+str(counter1+1)) * self.hidden_layer[another+1].output + \
                            self.hidden_layer[another+2].connections.get("output_neuron_"+str(counter1+1)) * self.hidden_layer[another+2].output + \
                            self.hidden_layer[another+3].connections.get("output_neuron_"+str(counter1+1)) * self.hidden_layer[another+3].output

                total += self.bias_for_output_layer.connections.get("bias_output_neuron_"+str(counter1+1)) * 1
                jj = self.output_layer[i].sigmoid(total)
                self.output_layer[i].output = jj
                counter1 += 1

    def back_propogation(self, target_output_1, target_output_2, target_output_3):
        '''

        :param target_output_1:
        :param target_output_2:
        :param target_output_3:
        :return:
        '''

        global hidden_layer
        global output_layer
        error_delta_hidden_neurons_ = []
        ### Calculation of the error for the output layer

        error_delta_output_neuron_1 = self.output_layer[0].output * (1 - self.output_layer[0].output) * (target_output_1 - self.output_layer[0].output)
        error_delta_output_neuron_2 = self.output_layer[1].output * (1 - self.output_layer[1].output) * (target_output_2 - self.output_layer[1].output)
        error_delta_output_neuron_3 = self.output_layer[2].output * (1 - self.output_layer[2].output) * (target_output_3 - self.output_layer[2].output)

        NN.temp_list = copy.deepcopy(self.hidden_layer)     ### used to calculate the delta of hidden layer

        ### Updating the weights of the hidden_to_output layer connections
        self.update_weights_of_hidden_and_output(error_delta_output_neuron_1, error_delta_output_neuron_2,
                                                 error_delta_output_neuron_3)

        ### Calculation of the error for the hidden layer
        COUNT = 0
        for i in range(0,4):

            #for j in range(1, 4):
            if COUNT < 4:
                temp = (self.hidden_layer[i].output \
                                      * (1 - self.hidden_layer[i].output) * ((error_delta_output_neuron_1 * NN.temp_list[i].connections.get('output_neuron_'+str(COUNT+1))) +
                                                                             (error_delta_output_neuron_2 * NN.temp_list[i].connections.get('output_neuron_'+str(COUNT+2))) +
                                                                             (error_delta_output_neuron_3 * NN.temp_list[i].connections.get('output_neuron_'+str(COUNT+3)))))
            error_delta_hidden_neurons_.append(temp)


        self.update_weights_of_input_and_hidden(error_delta_hidden_neurons_[0],error_delta_hidden_neurons_[1],error_delta_hidden_neurons_[2],error_delta_hidden_neurons_[3])

    def update_weights_of_hidden_and_output(self, error_delta_1, error_delta_2, error_delta_3):
        '''
        This function calculates the weights of the connection between hidden and the output layer, of which it has to be updated
        :param error_delta_1:   Error of the first output neuron
        :param error_delta_2:   Error of the second output neuron
        :param error_delta_3:   Error of the third output neuron
        :return:    None
        '''

        learning_factor = 0.2

        list_of_delta_errors = [error_delta_1,error_delta_2,error_delta_3]
        for i in range(0, 4):
            for j in range(1, 4):

                self.hidden_layer[i].connections['output_neuron_'+str(j)] = self.hidden_layer[i].connections.get('output_neuron_'+str(j)) \
                                                                            + learning_factor * list_of_delta_errors[j-1] * self.hidden_layer[i].output

        self.bias_for_output_layer.connections['bias_output_neuron_1'] = self.bias_for_output_layer.connections.get('bias_output_neuron_1') + learning_factor * error_delta_1 * 1
        self.bias_for_output_layer.connections['bias_output_neuron_2'] = self.bias_for_output_layer.connections.get('bias_output_neuron_2') + learning_factor * error_delta_2 * 1
        self.bias_for_output_layer.connections['bias_output_neuron_3'] = self.bias_for_output_layer.connections.get('bias_output_neuron_3') + learning_factor * error_delta_3 * 1

    def update_weights_of_input_and_hidden(self, error_delta_1, error_delta_2, error_delta_3, error_delta_4):
        '''
       This function calculates the weights of the connection between input and the hidden layer, of which it has to be updated
        :param error_delta_1:   Error of the first output neuron
        :param error_delta_2:   Error of the second output neuron
        :param error_delta_3:   Error of the third output neuron
        :return:    None
        '''

        learning_factor_2 = 0.2
        list_of_error_delta_input_and_hidden = [error_delta_1,error_delta_2,error_delta_3,error_delta_4]
        for i in range(0, 3):
            for j in range(1, 5):
                self.input_layer[i].connections['hidden_neuron_'+str(j)] = self.input_layer[i].connections.get('hidden_neuron_'+str(j)) \
                                                                           + learning_factor_2 * list_of_error_delta_input_and_hidden[j-1] * NN.list_of_inputs[i]

        self.bias_for_hidden_layer.connections['bias_hidden_neuron_1'] = self.bias_for_hidden_layer.connections.get('bias_hidden_neuron_1') + learning_factor_2 * error_delta_1 * 1
        self.bias_for_hidden_layer.connections['bias_hidden_neuron_2'] = self.bias_for_hidden_layer.connections.get('bias_hidden_neuron_2') + learning_factor_2 * error_delta_2 * 1
        self.bias_for_hidden_layer.connections['bias_hidden_neuron_3'] = self.bias_for_hidden_layer.connections.get('bias_hidden_neuron_3') + learning_factor_2 * error_delta_3 * 1
        self.bias_for_hidden_layer.connections['bias_hidden_neuron_4'] = self.bias_for_hidden_layer.connections.get('bias_hidden_neuron_4') + learning_factor_2 * error_delta_4 * 1




    def start(self,file_name):
        counter =0
        chara_1 = 'TH'
        chara_2 = 'HE'
        chara_3 = 'AN'
        chara_4 = 'EN'
        chara_5 = 'ER'
        chara_6 = 'EE'
        chara_7 = 'TO'
        chara_8 = 'RE'
        chara_9 = 'ON'


        with open(file_name, 'r') as f:

            for line in f:
                if not line:
                    continue
                else:

                    lower_count11 = line.count(chara_1.lower())
                    upper_count11 = line.count(chara_1)
                    mid_count11 = line.count('Th')
            
                    lower_count12 = line.count(chara_2.lower())
                    upper_count12 = line.count(chara_2)
                    mid_count12 = line.count('He')
            
            
                    lower_count13 = line.count(chara_3.lower())
                    upper_count13 = line.count(chara_3)
                    mid_count13 = line.count('An')
            
            
                    sum = lower_count11 + upper_count11 + mid_count11 + lower_count12+upper_count12+mid_count12 + lower_count13+upper_count13+mid_count13
                    #print(sum)
                    sum = (sum/(len(line) - line.count(' ')))*100
            
            
                    lower_count21 = line.count(chara_4.lower())
                    upper_count21 = line.count(chara_4)
                    mid_count21 = line.count('En')
            
            
            
                    lower_count22 = line.count(chara_5.lower())
                    upper_count22 = line.count(chara_5)
                    mid_count22 = line.count('Er')
            
            
                    lower_count23 = line.count(chara_6.lower())
                    upper_count23 = line.count(chara_6)
                    mid_count23 = line.count('Ee')
            
            
            
                    sum2 = lower_count21 + upper_count21 + mid_count21 + lower_count22 + upper_count22 + mid_count22 + lower_count23 + upper_count23 + mid_count23
                    #print("beofre ",sum2)
                    sum2 = (sum2/(len(line) - line.count(' ')))*100
            
                    lower_count31 = line.count(chara_7.lower())
                    upper_count31 = line.count(chara_7)
                    mid_count31 = line.count('To')
            
            
            
                    lower_count32 = line.count(chara_8.lower())
                    upper_count32 = line.count(chara_8)
                    mid_count32 = line.count('Re')
            
            
            
                    lower_count33 = line.count(chara_9.lower())
                    upper_count33 = line.count(chara_9)
                    mid_count33 = line.count('On')
            
            
            
                    sum3 = lower_count31 + upper_count31 + mid_count31 + lower_count32 + upper_count32 + mid_count32 + lower_count33 + upper_count33 + mid_count33
                    sum3 = (sum3/(len(line) - line.count(' ')))*100
            
            
                    self.feed_forward(sum,sum2,sum3)


n = NN(4)
x = raw_input("Enter the filename")

n.start(x)



if n.output_layer[0].output > n.output_layer[1].output and n.output_layer[0].output > n.output_layer[2].output:
    print ("english")


elif n.output_layer[2].output > n.output_layer[0].output and n.output_layer[2].output > n.output_layer[1].output:
    print("italian")
    
elif n.output_layer[1].output > n.output_layer[0].output and n.output_layer[1].output > n.output_layer[2].output:
    print("dutch")
















