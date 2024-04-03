import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import OneHotEncoder
import scipy.optimize
from tqdm import tqdm
import sklearn.decomposition

# For this assignment, assume that every hidden layer has the same number of neurons.
NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 10
NUM_OUTPUT = 10

# Unpack a list of weights and biases into their individual np.arrays.
def unpack (weightsAndBiases):
    """ Unpack a list of weights and biases into their individual np.arrays.

    Args:
        weightsAndBiases (list): List of weights and biases

    Returns:
        Ws (list): List of weight matrices
        bs (list): List of bias vectors
    """

    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT*NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN*NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN*NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs


def accuracy(y,y_hat):
    """ Calculate the accuracy of the model

    Args:
        y (np.ndarray): True labels
        y_hat (np.ndarray): Predicted labels

    Returns:
        float: Percentage of correctly classified samples
    """

    predicted_labels = np.argmax(y_hat.T, axis=1) #Returns the indices of the maximum values along column axis of y_hat
    output_labels = np.argmax(y, axis=1)

    correct = np.sum(predicted_labels==output_labels)
    percent_correct = correct/len(predicted_labels) #calculate percentage correctly classified
    return 100*percent_correct


def softmax(logits):
    """ Calculate the softmax of the logits

    Args:
        logits (np.ndarray): Logits

    Returns:
        np.ndarray: Softmax of the logits
    """

    return np.exp(logits) / np.sum(np.exp(logits), axis=0, keepdims=True)


def compute_ce_loss(y_label, y_pred):
    """ Compute the cross entropy loss

    Args:
        y_label (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels

    Returns:
        float: Cross entropy loss
    """

    loss = np.sum(-y_label * np.log(y_pred)) / y_label.shape[1]
    return loss


def forward_prop (x, y, weightsAndBiases):
    """ Forward propagate the inputs through the network

    Args:
        x (np.ndarray): Inputs
        y (np.ndarray): True labels
        weightsAndBiases (list): List of weights and biases

    Returns:
        loss (float): Cross entropy loss
        zs (list): List of pre-activations
        hs (list): List of post-activations
        yhat (np.ndarray): Predicted labels
    """

    Ws, bs = unpack(weightsAndBiases)

    zs = []
    hs = []

    input = x
    for i in range(NUM_HIDDEN_LAYERS):
        """
        _z = (10 x 784) @ (784 x 16) + (10, 1)
        _z shape = (10 x 16)
        """
        _z = Ws[i] @ input + bs[i].reshape(-1, 1)
        zs.append(_z)
        hs.append(np.maximum(0, _z))
        input = hs[-1]

    _z = Ws[-1] @ hs[-1] + bs[-1].reshape(-1, 1)
    zs.append(_z)
    yhat = softmax(_z) # 10 x 16

    loss = compute_ce_loss(y, yhat)

    # Return loss, pre-activations, post-activations, and predictions
    return loss, zs, hs, yhat


def relu(x):
    """ ReLU activation function

    Args:
        x (np.ndarray): Inputs

    Returns:
        np.ndarray: ReLU of the inputs
    """

    relu = x.copy()
    relu[relu < 0] = 0
    return relu


def relu_grad(x):
    """ Gradient of the ReLU activation function

    Args:
        x (np.ndarray): Inputs

    Returns:
        np.ndarray: Gradient of the ReLU of the inputs
    """

    grad = np.zeros_like(x)
    grad[x > 0] = 1
    return grad


def back_prop (x, y, weightsAndBiases):
    """ Back propagate the error through the network

    Args:
        x (np.ndarray): Inputs
        y (np.ndarray): True labels
        weightsAndBiases (list): List of weights and biases

    Returns:
        np.ndarray: Gradients of the weights and biases
    """

    Ws, bs = unpack(weightsAndBiases)

    loss, zs, hs, yhat = forward_prop(x, y, weightsAndBiases)

    dJdWs = []  # Gradients w.r.t. weights
    dJdbs = []  # Gradients w.r.t. biases

    delta = yhat - y # 10 x batch_size(16)

    for i in range(NUM_HIDDEN_LAYERS, -1, -1):

        if i != NUM_HIDDEN_LAYERS:
            fprime = relu_grad(zs[i])
            delta = fprime * delta

        if i == 0:
            dJdW = (delta @ x.T) / (y.shape[1])
        else:
            dJdW = (delta @ hs[i-1].T) / (y.shape[1])

        dJdb = np.sum(delta, axis=1)/y.shape[1]
        dJdbs.append(dJdb)

        dJdWs.append(dJdW)

        delta = Ws[i].T @ delta

    dJdWs.reverse()
    dJdbs.reverse()
    # Concatenate gradients
    return np.hstack([ dJdW.flatten() for dJdW in dJdWs ] + [ dJdb.flatten() for dJdb in dJdbs ])


def train (trainX, trainY, weightsAndBiases, testX, testY):
    """ Train the network

    Args:
        trainX (np.ndarray): Training data
        trainY (np.ndarray): Training labels
        weightsAndBiases (list): List of weights and biases
        testX (np.ndarray): Test data
        testY (np.ndarray): Test labels

    Returns:
        weightsAndBiases (list): List of weights and biases
        trajectory (list): List of weights and biases during training
    """

    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNINGRATE = 0.01
    trajectory = []

    num_samples = trainX.shape[1]
    indices = np.arange(num_samples)

    iterations = trainX.shape[1] / BATCH_SIZE

    for epoch in tqdm(range(NUM_EPOCHS)):
        # TODO: implement SGD.
        # TODO: save the current set of weights and biases into trajectory; this is
        # useful for visualizing the SGD trajectory.

        # Shufflling the indices
        np.random.shuffle(indices)

        trainX_shuffled = trainX[:, indices]
        trainY_shuffled = trainY[:, indices]


        for iter in range(int(iterations)):
            """
            Size:
            m x batch_size --> 784 x batch_size
            """
            X_batch = trainX_shuffled[:, (iter * BATCH_SIZE) : (iter + 1) * BATCH_SIZE]
            """
            Size:
            10 x batch_size
            """
            y_batch = trainY_shuffled[:, (iter * BATCH_SIZE) : (iter + 1) * BATCH_SIZE]
            gradients = back_prop(X_batch, y_batch, weightsAndBiases)

            weightsAndBiases -= LEARNINGRATE * gradients

            trajectory.append(weightsAndBiases.copy())

    return weightsAndBiases, trajectory


# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases ():
    """ Performs a standard form of random initialization of weights and biases

    Args:
        None

    Returns:
        np.ndarray: List of weights and biases
    """

    Ws = []
    bs = []

    np.random.seed(0)
    W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2*(np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([ W.flatten() for W in Ws ] + [ b.flatten() for b in bs ])

def plotSGDPath (trainX, trainY, trajectory):
    """ Plot the SGD trajectory

    Args:
        trainX (np.ndarray): Training data
        trainY (np.ndarray): Training labels
        trajectory (list): List of weights and biases during training

    Returns:
        None
    """

    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed 
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.

    trainX = trainX[:, :2500]
    trainY = trainY[:, :2500]
    def loss_function(weightsAndBiases):
        loss, _, _, _ = forward_prop(trainX, trainY, weightsAndBiases)
        return loss

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    sklearn_pca = sklearn.decomposition.PCA(n_components=2)
    fit_trajectory = sklearn_pca.fit_transform(trajectory)
    _ax1 = fit_trajectory[:, 0]
    _ax2 = fit_trajectory[:, 1]
    axis1 = np.linspace(np.min(_ax1), np.max(_ax1), 20)
    axis2 = np.linspace(np.min(_ax2), np.max(_ax2), 20)
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in tqdm(range(len(axis1))):
        for j in range(len(axis2)):
            _weightsAndBiases = sklearn_pca.inverse_transform(np.array([Xaxis[i,j], Yaxis[i,j]]))
            Zaxis[i,j] = loss_function(_weightsAndBiases)

    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('Loss')
    ax.set_title('SGD trajectory projected onto Loss surface')

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = []
    Yaxis = []
    Zaxis = []
    for i in tqdm(range(len(fit_trajectory))):
        if np.random.rand() < 0.005:
            x,y=fit_trajectory[i]
            _weightsAndBiases = trajectory[i]
            Zaxis.append(loss_function(_weightsAndBiases))
            Xaxis.append(x)
            Yaxis.append(y)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.savefig('SGD_path.png', dpi=1200, bbox_inches='tight')
    plt.close()


def train_data_split(trainX, tarinY, val_split_ratio):
    """ Split the dataset into train and validation sets

    Args:
        trainX (np.ndarray): Training data
        tarinY (np.ndarray): Training labels
        val_split_ratio (float): Ratio of validation data to training data

    Returns:
        X_train (np.ndarray): Training data
    """
    np.random.seed(11)

    # Shuffle the datatset
    data_indices = np.arange(len(trainX))
    np.random.shuffle(data_indices)
    val_data_num = int(len(trainX) * val_split_ratio)

    # Split the data in train and validation sets
    X_train = trainX[data_indices[val_data_num:]].T
    Y_train = tarinY[data_indices[val_data_num:]].reshape(-1, 1)
    X_val = trainX[data_indices[:val_data_num]].T
    y_val = tarinY[data_indices[:val_data_num]].reshape(-1, 1)

    return X_train, Y_train, X_val, y_val


if __name__ == "__main__":
    # Load data and split into train, validation, test sets
    X_tr = np.load("data/fashion_mnist_train_images.npy")/255.0 - 0.5
    y_tr = np.load("data/fashion_mnist_train_labels.npy")
    testX = np.load("data/fashion_mnist_test_images.npy")/255.0 - 0.5
    testY = np.load("data/fashion_mnist_test_labels.npy")

    # Split the dataset in train and val
    trainX, trainY, ValX, ValY = train_data_split(
        X_tr, y_tr, val_split_ratio=0.2
    )

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    trainY_encoded = encoder.fit_transform(trainY).T
    testY_encoded = encoder.fit_transform(testY.reshape(-1, 1)).T

    # Initialize weights and biases randomly
    weightsAndBiases = initWeightsAndBiases()

    Ws, bs = unpack(weightsAndBiases)

    # Perform gradient check on 5 training examples
    print('########## GRADIENT CHECK ############')
    print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY_encoded[:,0:5]), wab)[0], \
                                    lambda wab: back_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY_encoded[:,0:5]), wab), \
                                    weightsAndBiases))
    print('########## GRADIENT CHECK ############')

    # Train the network and obtain the SGD trajectory
    weightsAndBiases, trajectory = train(trainX, trainY_encoded, weightsAndBiases, testX, testY_encoded)

    # Plot the SGD trajectory
    plotSGDPath(trainX, trainY_encoded, trajectory)