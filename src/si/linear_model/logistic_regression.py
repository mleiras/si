import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
import numpy as np

from data.dataset import Dataset
from metrics.accuracy import accuracy
from statistics.sigmoid_function import sigmoid_function


class LogisticRegression:
    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 2000):
        """

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        alpha: float
            The learning rate
        max_iter: int
            The maximum number of iterations
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}

    def fit(self, dataset: Dataset) -> 'LogisticRegression':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: LogisticRegression
            The fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)
        self.theta_zero = 0

        # gradient descent
        for i in range(self.max_iter):
            # predicted y
            y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)

            # computing and updating the gradient with the learning rate
            gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y, dataset.X)

            # computing the penalty
            penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

            # updating the model parameters
            self.theta = self.theta - gradient - penalization_term
            self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

            self.cost_history[i] = round(self.cost(dataset),2)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        
        mask = y_pred >= 0.5
        y_pred[mask] = 1
        y_pred[~mask] = 0
        return y_pred
        

    def score(self, dataset: Dataset) -> float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Compute the cost function (J function) of the model on the dataset using L2 regularization

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the cost function on

        Returns
        -------
        cost: float
            The cost function of the model
        """
        y_pred = sigmoid_function(np.dot(dataset.X, self.theta) + self.theta_zero)
        cost = (-dataset.y*np.log(y_pred)) - ((1-dataset.y)*np.log(1-y_pred))
        cost = np.sum(cost) / dataset.shape()[0]
        cost = cost + (self.l2_penalty*np.sum(self.theta**2) / (2*dataset.shape()[0]))
        return cost


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # import dataset
    from data.dataset import Dataset
    from model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # fit the model
    model = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)
    model.fit(dataset_train)

    # compute the score
    score = model.score(dataset_test)
    print(f"Score: {score}")

    plt.plot(model.cost_history.values())

    plt.title("Cost History")
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.show()


    print('----------------------------------')


    from io_folder.module_csv import read_csv 
    breast_bin = read_csv('/home/monica/Documents/2_ano/sistemas/si/datasets/breast-bin.csv', sep=',',features=True, label=True)

    # fit the model    
    model = LogisticRegression()
    model.fit(breast_bin)

    # get coefs
    print(f"Parameters: {model.theta}")

    # compute the score
    score = model.score(breast_bin)
    print(f"Score: {score}")

    # compute the cost
    cost = model.cost(breast_bin)
    print(f"Cost: {cost}")

    # predict
    y_pred_ = model.predict(breast_bin)
    print(f"Predictions: {y_pred_}")

    # print(model.cost_history)

    plt.plot(model.cost_history.values())

    plt.title("Cost History")
    plt.ylabel("Cost")
    plt.xlabel("Iterations")
    plt.show()


    
