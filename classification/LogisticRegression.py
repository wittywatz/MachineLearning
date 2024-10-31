import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegression:
    def __init__(
        self, learning_rate=0.01, num_iterations=1000, weights=None, bias=None
    ):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = weights
        self.bias = bias

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, n_features):
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

    def compute_gradients(self, X, y, m):
        """
        Computes the gradients for logistic regression.

        Parameters:
        X (numpy.ndarray): Input features of shape (m, n_features).
        y (numpy.ndarray): Target values of shape (m, 1).
        m (int): Number of training examples.

        Returns:
        dw (numpy.ndarray): Gradient of the cost function with respect to weights of shape (n_features, 1).
        db (float): Gradient of the cost function with respect to bias.
        """
        Z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(Z)
        dz = A - y
        dw = (1 / m) * np.dot(X.T, dz)
        db = (1 / m) * np.sum(dz)
        return dw, db

    def compute_cost(self, X, y, m):
        """
        Computes the cost function for logistic regression.

        Parameters:
        X (numpy.ndarray): Input features of shape (m, n_features).
        y (numpy.ndarray): Target values of shape (m, 1).

        Returns:
        float: Cost function value.
        """
        Z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(Z)
        cost = -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
        return cost

    def fit(self, X, y):
        """
        Performs logistic regression training using gradient descent.

        Parameters:
        X (numpy.ndarray): Input features of shape (m, n_features).
        y (numpy.ndarray): Target values of shape (m, 1).
        """
        m, n_features = X.shape
        self.initialize_parameters(n_features)

        for _ in range(self.num_iterations):
            dw, db = self.compute_gradients(X, y, m)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print cost every 100 iterations
            if _ % 100 == 0:
                cost = self.compute_cost(X, y, m)
                print(f"Cost after iteration {_}: {cost}")

    def predict(self, X):
        """
        Predicts the class labels for input features.

        Parameters:
        X (numpy.ndarray): Input features of shape (m, n_features).

        Returns:
        numpy.ndarray: Predicted class labels of shape (m, 1).
        """
        Z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(Z)
        y_pred = (A >= 0.5).astype(int)
        return y_pred

    def evaluate(self, X, y):
        """
        Evaluates the model's performance using accuracy.

        Parameters:
        X (numpy.ndarray): Input features of shape (m, n_features).
        y (numpy.ndarray): Target values of shape (m, 1).

        Returns:
        float: Accuracy of the model.
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def save_model(self, filename):
        """
        Saves the trained logistic regression model to a file.

        Parameters:
        filename (str): Name of the file to save the model.
        """
        np.savez(filename, weights=self.weights, bias=self.bias)

    @staticmethod
    def load_model(filename):
        """
        Loads a trained logistic regression model from a file.

        Parameters:
        filename (str): Name of the file to load the model.

        Returns:
        LogisticRegression: Trained logistic regression model.
        """
        model_data = np.load(filename)
        return LogisticRegression(
            weights=model_data["weights"], bias=model_data["bias"]
        )


# Load the Iris dataset

iris = load_iris()
X = iris.data
y = iris.target

binary_class_indices = y != 2

# Select only the first two features and the class 0 and 1
X = X[binary_class_indices]
y = y[binary_class_indices]

# Convert labels to 0 and 1
y = y.reshape(y.shape[0], 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)

# Evaluate
train_accuracy = model.evaluate(X_train, y_train)
test_accuracy = model.evaluate(X_test, y_test)

print(f"Train accuracy: {train_accuracy:.2f}")
print(f"Test accuracy: {test_accuracy:.2f}")

# # Save the model
model.save_model("saved_models/logistic_regression_model.npz")

# # Load the saved model
loaded_model = LogisticRegression.load_model(
    "saved_models/logistic_regression_model.npz"
)

loaded_model_accuracy = loaded_model.evaluate(X_test, y_test)
print(f"Loaded model accuracy: {loaded_model_accuracy:.2f}")
