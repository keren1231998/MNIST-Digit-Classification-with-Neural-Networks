from time import time

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def initialize_parameters(layer_dims):
    """
        Initialize the parameters for each layer in the neural network.

        Parameters:
        - layer_dims (list of int): Dimensions of each layer in the network.

        Returns:
        - parameters (dict): Dictionary containing the initialized weights and biases for each layer.
    """
    np.random.seed(42)  # Seed for reproducibility
    parameters = {}
    for l in range(1, len(layer_dims)):
        red_param =np.sqrt(1 / layer_dims[l - 1])

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * red_param
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    """
        Implement the linear part of a layer's forward propagation.

        Parameters:
        - A (ndarray): Activations from the previous layer (or input data).
        - W (ndarray): Weights matrix.
        - b (ndarray): Bias vector.

        Returns:
        - Z (ndarray): The input of the activation function (pre-activation parameter).
        - linear_cache (dict): A dictionary containing "A", "W", and "b" to store for computing the backward pass efficiently.
    """
    Z = np.dot(W, A) + b
    linear_cache = {"A": A, "W": W, "b": b}
    return Z, linear_cache

def softmax(Z):
    """
        Compute the softmax activation.

        Args:
        Z (np.ndarray): Linear output.

        Returns:
        tuple: Softmax activation A and input Z.
    """
    shift_z = Z - np.max(Z, axis=0, keepdims=True)
    exp_z = np.exp(shift_z)
    A = exp_z / np.sum(exp_z, axis=0, keepdims=True)
    return A, Z

def relu(Z):
    """
        Compute the ReLU activation.

        Args:
        Z (np.ndarray): Linear output.

        Returns:
        tuple: ReLU activation A and input Z.
    """
    A = np.maximum(0, Z)
    return A, Z

def linear_activation_forward(A_prev, W, b, activation):
    """
         Perform forward propagation for a single layer with a specified activation function.

         Args:
         A_prev (np.ndarray): Activations from the previous layer.
         W (np.ndarray): Weights matrix.
         b (np.ndarray): Bias vector.
         activation (str): Activation function to use ("relu" or "softmax").

         Returns:
         tuple: Activation output A and cache dictionary.
     """
    Z, linear_dict = linear_forward(A_prev, W, b)
    if activation == "softmax":
        A, activation_cache = softmax(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    linear_dict["Z"] = Z

    return A, linear_dict

def L_model_forward(X, parameters, use_batchnorm=False):
    """
      Perform forward propagation for the entire network.

      Args:
      X (np.ndarray): Input data.
      parameters (dict): Initialized parameters.
      use_batchnorm (bool): Whether to apply batch normalization.

      Returns:
      tuple: Output of the last layer AL and list of caches.
    """
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "softmax")
    caches.append(cache)
    return AL, caches

def load_and_preprocess_data():
    """
        Load and preprocess the MNIST dataset.

        Returns:
        tuple: Preprocessed training, validation, and test sets.
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)

    return X_train, y_train, X_val, y_val, X_test, y_test

def relu_backward(dA, cache):
    """
       Perform backward propagation for a ReLU activation.

       Args:
       dA (np.ndarray): Gradient of the cost with respect to the activation.
       cache (np.ndarray): Input Z from the forward pass.

       Returns:
       np.ndarray: Gradient of the cost with respect to Z.
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax_backward(dA, cache):
    """
       Perform backward propagation for a softmax activation.

       Args:
       dA (np.ndarray): Gradient of the cost with respect to the activation.
       cache (np.ndarray): Input Y from the forward pass.

       Returns:
       np.ndarray: Gradient of the cost with respect to Z.
    """
    Y = cache
    dZ = np.subtract(dA, Y)
    return dZ


def linear_backward(dZ, cache):
    """
       Perform the linear part of backward propagation.

       Args:
       dZ (np.ndarray): Gradient of the cost with respect to Z.
       cache (dict): Dictionary containing A, W, b from the forward pass.

       Returns:
       tuple: Gradients of the cost with respect to A_prev, W, b.
    """
    A_prev, W, b = cache['A'], cache['W'], cache['b']
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
       Perform backward propagation for a single layer with a specified activation function.

       Args:
       dA (np.ndarray): Gradient of the cost with respect to the activation.
       cache (dict): Cache dictionary from the forward pass.
       activation (str): Activation function used ("relu" or "softmax").

       Returns:
       tuple: Gradients of the cost with respect to A_prev, W, b.
    """
    if activation == "relu":
        activation_cache = cache['Z']
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        activation_cache = cache['Y']
        dZ = softmax_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
        Perform backward propagation for the entire network.

        Args:
        AL (np.ndarray): Output of the last layer.
        Y (np.ndarray): True labels.
        caches (list): List of caches from the forward pass.

        Returns:
        dict: Gradients of the cost with respect to all parameters.
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = AL
    current_cache = caches[-1]
    current_cache['Y'] = Y
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "softmax")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+2)], current_cache, "relu")
        grads["dA" + str(l+1)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
    return grads

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size,
                  validation_X=None, validation_Y=None, lambd=0.01, L2=False, early_stopping_patience=100, validation_start_epoch=25,use_batchnorm=False):
    """
      Train a L-layer neural network.

      Args:
      X (np.ndarray): Training input data.
      Y (np.ndarray): Training labels.
      layers_dims (list): Dimensions of each layer in the network.
      learning_rate (float): Learning rate for gradient descent.
      num_iterations (int): Number of iterations for training.
      batch_size (int): Size of mini-batches.
      validation_X (np.ndarray, optional): Validation input data.
      validation_Y (np.ndarray, optional): Validation labels.
      L2 (bool, optional): Whether to apply L2 regularization.
      early_stopping_patience (int, optional): Number of iterations to wait for improvement before stopping.
      validation_start_epoch (int, optional): Epoch to start validation.
      use_batchnorm (bool, optional): Whether to apply batch normalization.

      Returns:
      tuple: Best parameters found during training, list of costs, list of accuracies.
    """
    accuracies=[]
    start_time = time()
    costs = []
    validation_costs = []
    parameters = initialize_parameters(layers_dims)

    training_step = 0
    stop_counter = 0
    num_epochs = 0
    best_parameters = parameters
    best_cost = float('inf')

    num_examples = X.shape[1]
    print(num_examples)

    for i in range(num_iterations):
        num_epochs += 1

        # Shuffle the dataset
        indices = np.arange(num_examples)
        np.random.shuffle(indices)
        X_shuffled = X[:, indices]
        Y_shuffled = Y[:, indices]

        for j in range(0, num_examples, batch_size):
            X_batch = X_shuffled[:, j:j + batch_size]
            Y_batch = Y_shuffled[:, j:j + batch_size]

            AL, caches = L_model_forward(X_batch, parameters, use_batchnorm=use_batchnorm)
            cost = compute_cost(AL, Y_batch, L2, parameters, lambd)
            grads = L_model_backward(AL, Y_batch, caches)
            parameters = update_parameters(parameters, grads, learning_rate, L2, lambd)

            # Print cost every 100 iterations
            if training_step % 100 == 0:
                costs.append(cost)
                val_accuracy = predict(validation_X,  np.argmax(validation_Y, axis=1), parameters)
                print(f"training_step {training_step}: Cost: {cost:.4f},accuracy: {val_accuracy:.4f}")

                accuracies.append(val_accuracy)

            training_step += 1

            # Validate model every 100 steps after the initial 25 epochs
            if validation_X is not None and validation_Y is not None and i > validation_start_epoch:
                AL, caches = L_model_forward(validation_X, parameters)
                val_cost = compute_cost(AL, validation_Y.T, L2, parameters, lambd)

                validation_costs.append(val_cost)

                if best_cost is not None and val_cost < best_cost:
                    best_cost = val_cost
                    best_parameters = parameters
                    stop_counter = 0
                elif best_cost is None:
                    best_cost = val_cost
                    best_parameters = parameters
                else:
                    stop_counter += 1

                if stop_counter >= early_stopping_patience:
                    print(f"No improvement for {early_stopping_patience} steps. Stopping early.")
                    break

        if stop_counter >= early_stopping_patience:
            break

    print(f"Total epochs: {num_epochs}")
    print(f"Training the network took {time() - start_time:.2f} seconds")

    return best_parameters, costs,accuracies


def update_parameters(parameters, grads, learning_rate, L2=False, lambd=0.01):
    """
        Update parameters using gradient descent with optional L2 regularization.

        Args:
        parameters (dict): Current parameters.
        grads (dict): Gradients of the cost with respect to parameters.
        learning_rate (float): Learning rate for gradient descent.
        L2 (bool, optional): Whether to apply L2 regularization.
        lambd (float, optional): Regularization parameter.

        Returns:
        dict: Updated parameters.
    """
    L = len(parameters) // 2
    m = grads["dA" + str(L)].shape[1]
    for l in range(1, L+1):
        if L2:
            parameters["W" + str(l)] -= learning_rate * (grads["dW" + str(l)] + (lambd / m) * parameters["W" + str(l)])
        else:
            parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters

def predict(X, Y,parameters,use_batchnorm=False):
    """
       Make predictions with the trained neural network.

       Args:
       X (np.ndarray): Input data.
       Y (np.ndarray): True labels.
       parameters (dict): Trained parameters.
       use_batchnorm (bool, optional): Whether to apply batch normalization.

       Returns:
       float: Accuracy of predictions.
    """
    probas, caches = L_model_forward(X, parameters, use_batchnorm=use_batchnorm)
    predictions = np.argmax(probas, axis=0)
    return np.mean(predictions == Y)



def compute_cost(AL, Y, L2=False, parameters=None, lambd=0.01):
    """
       Compute the cost function with optional L2 regularization.

       Args:
       AL (np.ndarray): Activation output of the last layer.
       Y (np.ndarray): True labels.
       L2 (bool, optional): Whether to apply L2 regularization.
       parameters (dict, optional): Parameters for regularization.
       lambd (float, optional): Regularization parameter.

       Returns:
       float: Cost value.
    """
    m = Y.shape[1]
    cross_entropy_cost = -np.sum(np.log(AL + 1e-8) * Y) / m

    if L2 and parameters:
        L2_cost = 0
        for l in range(1, len(parameters) // 2 + 1):
            L2_cost += np.sum(np.square(parameters['W' + str(l)]))
        L2_cost = (lambd / (2 * m)) * L2_cost
        cost = cross_entropy_cost + L2_cost
    else:
        cost = cross_entropy_cost

    return cost
def apply_batchnorm(A):
    """
        Perform batch normalization on the output of the activation layer.

        Args:
        A (np.ndarray): Post-activation output from a layer.

        Returns:
        np.ndarray: Batch normalized activation outputs.
    """
    mu = np.mean(A, axis=1, keepdims=True)
    var = np.var(A, axis=1, keepdims=True)

    A_norm = (A - mu) / np.sqrt(var + 1e-8)
    return A_norm


def plot_costs(costs):
    """
    Plot the cost values over iterations during the learning process of the model.

    Parameters:
    costs (list or numpy.ndarray): List of cost values at each iteration.

    Returns:
    None

    Note:
    - This function creates a line plot to visualize how the cost changes over training iterations.
    - It is helpful for monitoring the learning progress and convergence of the model.
    """
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning process of the model")
    plt.show()
def plot_accuracy(accuracy):
    """
    Plot the cost values over iterations during the learning process of the model.

    Parameters:
    costs (list or numpy.ndarray): List of cost values at each iteration.

    Returns:
    None

    Note:
    - This function creates a line plot to visualize how the cost changes over training iterations.
    - It is helpful for monitoring the learning progress and convergence of the model.
    """
    plt.plot(accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning process of the model")
    plt.show()

def print_network_weights(parameters):
        """
        Print all the weights and biases of the neural network.

        Args:
        parameters (dict): Dictionary containing the network parameters (weights and biases).

        Returns:
        None
        """
        for key, value in parameters.items():
            print(f"{key}: {value.shape}")
            print(value)


def part4():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

    layers_dims = [784, 20, 7, 5, 10]

    learning_rate = 0.009
    batch_size = 64
    stopping_criterion = 100
    parameters, costs ,accuracy= L_layer_model(X_train.T, y_train.T, layers_dims=layers_dims,
                                      learning_rate=learning_rate,
                                      num_iterations=stopping_criterion, batch_size=batch_size,
                                      validation_X=X_val.T, validation_Y=y_val, L2=False,use_batchnorm=False)
    plot_accuracy(accuracy)
    plot_costs(costs)
    train_accuracy = predict(X_train.T, np.argmax(y_train, axis=1), parameters, use_batchnorm=False)
    validation_accuracy = predict(X_val.T, np.argmax(y_val, axis=1), parameters, use_batchnorm=False)
    test_accuracy = predict(X_test.T, np.argmax(y_test, axis=1), parameters, use_batchnorm=False)
    print_network_weights(parameters)

    print("train_accuracy : {:.2%}".format(train_accuracy))
    print("validation_accuracy : {:.2%}".format(validation_accuracy))
    print("test_accuracy : {:.2%}".format(test_accuracy))

def part5():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

    layers_dims = [784, 20, 7, 5, 10]

    learning_rate = 0.009
    batch_size = 64
    stopping_criterion = 100
    parameters, costs ,accuracy= L_layer_model(X_train.T, y_train.T, layers_dims=layers_dims,
                                      learning_rate=learning_rate,
                                      num_iterations=stopping_criterion, batch_size=batch_size,
                                      validation_X=X_val.T, validation_Y=y_val, L2=False,use_batchnorm=True)
    plot_accuracy(accuracy)
    plot_costs(costs)
    train_accuracy = predict(X_train.T, np.argmax(y_train, axis=1), parameters, use_batchnorm=True)
    validation_accuracy = predict(X_val.T, np.argmax(y_val, axis=1), parameters, use_batchnorm=True)
    test_accuracy = predict(X_test.T, np.argmax(y_test, axis=1), parameters, use_batchnorm=True)
    print_network_weights(parameters)

    print("train_accuracy : {:.2%}".format(train_accuracy))
    print("validation_accuracy : {:.2%}".format(validation_accuracy))
    print("test_accuracy : {:.2%}".format(test_accuracy))
def part6():
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

    layers_dims = [784, 20, 7, 5, 10]

    learning_rate = 0.009
    batch_size = 64
    stopping_criterion = 100
    parameters, costs ,accuracy= L_layer_model(X_train.T, y_train.T, layers_dims=layers_dims,
                                      learning_rate=learning_rate,
                                      num_iterations=stopping_criterion, batch_size=batch_size,
                                      validation_X=X_val.T, validation_Y=y_val, L2=True,use_batchnorm=False)
    plot_accuracy(accuracy)
    plot_costs(costs)
    train_accuracy = predict(X_train.T, np.argmax(y_train, axis=1), parameters, use_batchnorm=False)
    validation_accuracy = predict(X_val.T, np.argmax(y_val, axis=1), parameters, use_batchnorm=False)
    test_accuracy = predict(X_test.T, np.argmax(y_test, axis=1), parameters, use_batchnorm=False)
    print_network_weights(parameters)

    print("train_accuracy : {:.2%}".format(train_accuracy))
    print("validation_accuracy : {:.2%}".format(validation_accuracy))
    print("test_accuracy : {:.2%}".format(test_accuracy))

if __name__ == "__main__":
    part4()
    # part5()
    # part6()
# ________________________________________
def experiment_with_batch_sizes(X_train, y_train, X_val, y_val, layers_dims, learning_rate, num_iterations, batch_sizes, early_stopping_patience=100, validation_start_epoch=25):
    batch_size_results = {}
    for batch_size in batch_sizes:
        print(f"Training with batch size: {batch_size}")
        parameters, costs, accuracy = L_layer_model(
            X_train.T, y_train.T, layers_dims=layers_dims,
            learning_rate=learning_rate,
            num_iterations=num_iterations, batch_size=batch_size,
            validation_X=X_val.T, validation_Y=y_val, L2=False,
            early_stopping_patience=early_stopping_patience,
            validation_start_epoch=validation_start_epoch
        )
        final_train_accuracy = predict(X_train.T, np.argmax(y_train, axis=1), parameters, use_batchnorm=False)
        final_validation_accuracy = predict(X_val.T, np.argmax(y_val, axis=1), parameters, use_batchnorm=False)
        batch_size_results[batch_size] = {
            "train_accuracy": final_train_accuracy,
            "validation_accuracy": final_validation_accuracy,
            "costs": costs,
            "accuracy": accuracy
        }
    return batch_size_results


# Run the experiments
# if __name__ == "__main__":
    # experiment_with_batch_sizes