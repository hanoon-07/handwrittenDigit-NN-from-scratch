import mnist_loader
from network import Network

# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Initialize network with 784 input neurons, 30 hidden neurons, and 10 output neurons
net = Network([784,10])

# Train network
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
