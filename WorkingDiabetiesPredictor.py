import numpy as np
import pandas


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
#feature_set = np.array(dataframe.values)

#Person	Smoking	Obesity	Exercise	Diabetic
#  1	    0     1        0        	1
#  2	    0     0        1        	0
#  3	    1     0        0        	0
#  4	    1     1        0        	1
#  5	    1     1        1        	1

feature_set = np.array([[0,1,0],
                        [0,0,1],
                        [1,0,0],
                        [1,1,0],
                        [1,1,1]]) #Sample data
labels = np.array([[1,
                    0,
                    0,
                    1,
                    1
                    ]]) # Answers

# Sample data could be the station alams (table with loads of bits in)
labels = labels.reshape(5,1)

np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

for epoch in range(100000):
    inputs = feature_set
    # feedforward step1
    XW = np.dot(feature_set, weights) + bias
    #feedforward step2
    z = sigmoid(XW)
    # backpropagation step 1
    error = z - labels
    print(error.sum())
    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num


while(True): 
    print("--------")
    inSmoker = float(input("Do you smoke? "))
    inObese = float(input("Do are you obese? "))
    inExercise = float(input("Do you exercise? "))

    single_point = np.array([inSmoker,inObese,inExercise])
    #single_point = np.array([0,0,1])
    result = sigmoid(np.dot(single_point, weights) + bias)
    resultPercent = result*100
    print("Chance of diabities: %.2f" % resultPercent + "%")