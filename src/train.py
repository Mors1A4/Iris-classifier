import argparse
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_model(test_size, random_state):
    iris = load_iris()
    X = iris.data
    y = iris.target
    print(f"Feature names: {iris.feature_names}")
    print(f"Catagories: {iris.target_names}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model= DecisionTreeClassifier(random_state=random_state)

    print("Training model...")
    model.fit(X_train, y_train)

    print("Testing model on unseen examples...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    print(f"Model tested on {len(y_pred)} examples with an accuracy of {accuracy}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_size",type=float, default=0.2, help="Proportion of the dataset to test on")

    parser.add_argument("--random_state", type=int, default=42, help="The seed used by the random number functions")

    args = parser.parse_args()

    train_model(args.test_size, args.random_state)

