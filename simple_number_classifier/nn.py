import numpy
import scipy.special


class NeuralNetwork:
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        
        # Set inital weights to random value in normal distribution
        self.wih = numpy.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.woh = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        
        # activation function is sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    

    def train(self, inputs_list, targets_list):
        # get 2D transpose of inputs and targets
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # 2nd layer (hidden layer inputs)
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 3rd layer (output layer inputs)
        final_inputs = numpy.dot(self.woh, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        # Get final output error
        output_errors = targets - final_outputs
        
        # Backpropagate errors by output of hidden weights
        hidden_errors = numpy.dot(self.woh.T, output_errors)
        
        # Update weights for hidden --> output layer links
        self.woh += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # Update weights for input --> hidden layer links
        self.wih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass
    

    def query(self, inputs_list):
        # get 2D transpose of inputs (1st layer - input layer)
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # 2nd layer (hidden layer)
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 3rd layer (output layer)
        final_inputs = numpy.dot(self.woh, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
