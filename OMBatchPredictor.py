import numpy as np
import pandas

dataCSVPath = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
#headders = ['preg', 'plas', 'pres', 'skin','test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(dataCSVPath)#, names=headders)
featureSet = np.array(dataframe.values)

finalSet = np.array([]) # Array of data/inputs
labels = np.array([]) # Array of answers

rowCount = 0
for row in featureSet:
    labels = np.append(labels, row[-1]) # Take the last value as the answer
    if rowCount == 0: # If it's first itteration then we can't vstack to an empty array
        finalSet = np.append(finalSet, np.array([row[:-1]]))
    else:
        finalSet = np.vstack((finalSet, np.array([row[:-1]]))) # Stack the arrays don't merge
    rowCount += 1

labels = labels.reshape(len(labels), 1)

np.random.seed(42)
weights = np.random.rand(len(finalSet[0]), 1)  # (4,1)
bias = np.random.rand(1)
lr = 0.05


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


for epoch in range(300):
    inputs = finalSet
    # feedforward step1
    XW = np.dot(finalSet, weights) + bias
    # feedforward step2
    z = sigmoid(XW)
    # backpropagation step 1
    error = z - labels
    print(error.sum())
    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = finalSet.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num


single_point = np.array([1, 0, 0, 0])
result = sigmoid(np.dot(single_point, weights) + bias)
resultPercent = result*100
print("Chance of pass: %.2f" % resultPercent + "%")


# print("--------")
# single_point = np.array([0,0,1,0])
# typeLabels = ["Smoker", "Obese", "Exercise", "Smoke&Obese", "Obest&Exercise", "Cure"]
# testCases = [np.array([1,0,0,0]), np.array([0,1,0,0]), np.array([0,0,1,0]), np.array([1,1,0,0]), np.array([0,1,1,0]), np.array([0,0,0,1])]

# caseLoop = 0
# for case in testCases:
#     result = sigmoid(np.dot(case, weights) + bias)
#     resultPercent = result*100
#     print(typeLabels[caseLoop] + ": %.2f" % resultPercent + "%")
#     caseLoop = caseLoop+1


# while(True):
#     print("--------")
#     inSmoker = float(input("Humidity: "))
#     inObese = float(input("Temp: "))
#     inExercise = float(input("Pressure: "))

#     single_point = np.array([inSmoker,inObese,inExercise])
#     result = sigmoid(np.dot(single_point, weights) + bias)
#     resultPercent = result*100
#     print("Chance of pass: %.2f" % resultPercent + "%")
