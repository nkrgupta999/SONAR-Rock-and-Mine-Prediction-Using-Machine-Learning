import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Importing the dataset
df = pd.read_csv("Sonar_data.csv", header=None)
print("Dataset: \n", df.head())

# Shape of dataset
print("Shape of dataset: \n", df.shape)

# Statistical measures of data
print("\n")
print("Statistical measures of data: \n", df.describe())
print("Information of dataset: \n", df.info())

# Check for null values using heatmap
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis")
plt.title("Null Values Heatmap")
plt.show()

# Distribution of classes in target column
print(df[60].value_counts())
print(df.groupby(60).mean())


# One hot encoding for the target value
def convertion(text):
    if "R" in text:
        return 0
    elif "M" in text:
        return 1


df[60] = df[60].apply(convertion)
print("Target after encoding: \n", df[60])

# Separating Data and Labels
X = df.drop(columns=60, axis=1)  # Independent data
Y = df[60]  # Dependent data

# Feature Importance
# Splitting the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
print(X_train.shape, X_test.shape)


# Model Evaluation on Train Data
def models_train(X_train, Y_train):
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)

    knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    knn.fit(X_train, Y_train)

    svc_lin = SVC(kernel="linear", random_state=0)
    svc_lin.fit(X_train, Y_train)

    svc_rbf = SVC(kernel="rbf", random_state=0)
    svc_rbf.fit(X_train, Y_train)

    gauss = GaussianNB()
    gauss.fit(X_train, Y_train)

    tree = DecisionTreeClassifier(criterion="entropy", random_state=42)
    tree.fit(X_train, Y_train)

    forest = RandomForestClassifier(
        n_estimators=10, criterion="entropy", random_state=0
    )
    forest.fit(X_train, Y_train)

    print("MACHINE LEARNING ALGORITHM APPLIED CHECK OUT ACCURACY ON TRAIN DATA")
    print("")
    print("[0] logistic regression Accuracy             :", log.score(X_train, Y_train))
    print("")
    print("[1]  K N N algorithm  Accuracy               :", knn.score(X_train, Y_train))
    print("")
    print(
        "[2] Support Vector Machine(svc) Accuracy     :",
        svc_lin.score(X_train, Y_train),
    )
    print("")
    print(
        "[3] Support Vector Machine(rbf)  algorithm   :",
        svc_rbf.score(X_train, Y_train),
    )
    print("")
    print(
        "[4] Naive Bayes Accuracy Accuracy            :", gauss.score(X_train, Y_train)
    )
    print("")
    print(
        "[5] Decision Tree Algorithm Accuracy         :", tree.score(X_train, Y_train)
    )
    print("")
    print(
        "[6] Random Forest Accuracy                   :", forest.score(X_train, Y_train)
    )
    print("")
    return log, knn, svc_lin, svc_rbf, gauss, tree, forest


model_train = models_train(X_train, Y_train)
print("model_train: ", model_train)


# Model Evaluation on Test Data
def models_test(X_test, Y_test):
    log = LogisticRegression(random_state=0)
    log.fit(X_test, Y_test)

    knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    knn.fit(X_test, Y_test)

    svc_lin = SVC(kernel="linear", random_state=0)
    svc_lin.fit(X_test, Y_test)

    svc_rbf = SVC(kernel="rbf", random_state=0)
    svc_rbf.fit(X_test, Y_test)

    gauss = GaussianNB()
    gauss.fit(X_test, Y_test)

    tree = DecisionTreeClassifier(criterion="entropy", random_state=42)
    tree.fit(X_test, Y_test)

    forest = RandomForestClassifier(
        n_estimators=10, criterion="entropy", random_state=0
    )
    forest.fit(X_test, Y_test)

    print("MACHINE LEARNING ALGORITHM APPLIED CHECK OUT ACCURACY ON TEST DATA")
    print("")
    print("[0] logistic regression Accuracy             :", log.score(X_test, Y_test))
    print("")
    print("[1]  K N N algorithm  Accuracy               :", knn.score(X_test, Y_test))
    print("")
    print(
        "[2] Support Vector Machine(svc) Accuracy     :", svc_lin.score(X_test, Y_test)
    )
    print("")
    print(
        "[3] Support Vector Machine(rbf)  algorithm   :", svc_rbf.score(X_test, Y_test)
    )
    print("")
    print("[4] Naive Bayes Accuracy Accuracy            :", gauss.score(X_test, Y_test))
    print("")
    print("[5] Decision Tree Algorithm Accuracy         :", tree.score(X_test, Y_test))
    print("")
    print(
        "[6] Random Forest Accuracy                   :", forest.score(X_test, Y_test)
    )
    print("")
    return log, knn, svc_lin, svc_rbf, gauss, tree, forest


model_test = models_test(X_test, Y_test)

# Making Prediction
input_data = (0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032,
)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_test, Y_test)

prediction = clf.predict(input_data_reshaped)
print(prediction)

if prediction[0] != 1:
    print("The object is a Rock")
else:
    print("The object is a mine")
# Confusion Matrix
# Make predictions on the test data
y_pred = clf.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(Y_test, y_pred)

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()