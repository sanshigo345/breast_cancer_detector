import numpy as np # 1.16.4
import pandas as pd # 0.24.2
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-0.005*x))

def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)

def read_and_divide_into_train_and_test(csv_file):
    cancer_data = pd.read_csv(csv_file)
    cancer_data.isnull().sum()
    cancer_data.drop("Code_number", axis=1, inplace=True)
    df = cancer_data.dropna(axis=0)
    df.index=range(0,len(df),1)
    df["Bare_Nuclei"] = df["Bare_Nuclei"][df["Bare_Nuclei"] != "?"]
    df.dropna(inplace=True)
    df["Bare_Nuclei"] = df["Bare_Nuclei"].astype("int64")

    train = df.sample(frac=0.8, random_state=100)
    training_inputs = train.iloc[:, :-1].values
    training_labels = train.iloc[:, -1].values
    training_labels = training_labels.reshape(training_labels.shape[0],-1)
    test = df.drop(train.index)
    test_inputs = test.iloc[:, :-1].values
    test_labels = test.iloc[:, -1].values

    label = train.columns
    correlations = train.corr()
    figure, ax = plt.subplots()
    figure.subplots_adjust(bottom=0.4, left=0.3)
    visualize = ax.pcolor(correlations)
    plt.colorbar(visualize)
    ticks = np.arange(correlations.shape[0])
    ax.set_xticks(ticks + 0.5, minor=False)
    ax.set_yticks(ticks + 0.5, minor=False)
    ax.set_xticklabels(label, rotation=90)
    ax.set_yticklabels(label, rotation=20)
    for i in range(len(label)):
        for j in range(len(label)):
            text = ax.text(j, i, correlations[i, j], ha="center", va="center", color="w")

    plt.show()

    return training_inputs, training_labels, test_inputs, test_labels

def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0
    predictions = sigmoid(np.dot(test_inputs, weights))

    for i in predictions:
        if i[0] <= 0.5:
            i[0] = 0
        else:
            i[0] = 1

    for predicted_val, label in zip(predictions, test_labels):
        if predicted_val == label:
            tp += 1

    accuracy = tp / len(predictions)    # accuracy = tp / total number of samples
    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array):
    plt.plot(loss_array)
    plt.title("Loss-Iteration Plot")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(accuracy_array)
    plt.title("Accuracy-Iteration Plot")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.show()

def main():
    csv_file = "breast-cancer-wisconsin.csv"
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):
        outputs = np.dot(training_inputs, weights)
        outputs = sigmoid(outputs)
        loss = training_labels - outputs
        tuning = loss * sigmoid_derivative(outputs)
        weights += np.dot(training_inputs.T, tuning)
        accuracy_array.append(run_on_test_set(test_inputs, test_labels, weights))
        loss_array.append((np.mean(loss)))

    plot_loss_accuracy(accuracy_array, loss_array)

if __name__ == '__main__':
    main()

