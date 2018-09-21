import numpy
from nn import NeuralNetwork


input_nodes = 784
hidden_nodes = 200
output_nodes = 10
    
learning_rate = 0.33

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Train on each training data point

# Read training data file
training_data_file = open("mnist_dataset/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()


## EPOCH TRAIN CHOO CHOO !!!
epochs = 2

# Train Neural Net multiple times
for x in range(epochs):
    
    # Train each record available
    for record in training_data_list:
        all_values = record.split(',')

        # Scale input between 0.01 and 1.00
        scaled_inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01

        # all_values[0] is the label for the records
        targets[int(all_values[0])] = 0.99
        n.train(scaled_inputs, targets)


# Test Neural Net on each test data point

# Read test data file
test_data_file = open("mnist_dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# Score Neural Net
scorecard = 0
total_test_inputs = len(test_data_list)


# Test each data point
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])

    # Scale input between 0.01 and 1.00
    scaled_inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    # Ask Neural Net for answer
    output = n.query(scaled_inputs)
    output_label = numpy.argmax(output)


    if (output_label == correct_label):
        scorecard += 1

print(str(scorecard / float(total_test_inputs) * 100) + ' % accuracy')