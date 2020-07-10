import numpy as np
import pandas


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
#feature_set = np.array(dataframe.values)

#Person	Smoking	Obesity	Exercise	 Anti-diabeties pill    Diabetic  
#  1	    0     1        0        	      1                 0              
#  2	    0     0        1        	      0                 0
#  3	    1     0        0        	      0                 0
#  4	    1     1        0        	      0                 1
#  5	    1     1        1        	      1                 0
#  5	    0     1        1        	      1                 0
#  5	    1     1        1        	      0                 1
#  5	    1     0        0        	      0                 1
#  5	    0     1        0        	      1                 0
#  5	    1     1        0        	      0                 1
#  5	    1     0        0        	      0                 1
#  5	    0     0        0        	      0                 0
#  5	    0     0        1        	      0                 0
#  5	    1     1        0        	      0                 1

feature_set = np.array([[0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [1, 1, 1, 1],
                        [0, 1, 1, 1],
                        [1, 1, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 1],
                        [1, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 1, 0],
                        [1, 1, 0, 0]
                        ]) #Sample data
labels = np.array([0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1
                    ]) # Answers

# Sample data could be the station alams (table with loads of bits in)
labels = labels.reshape(len(labels),1)

np.random.seed(42)
weights = np.random.rand(len(feature_set[0]),1) #(4,1)
bias = np.random.rand(1)
lr = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

for epoch in range(300):
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
        
print("--------")
single_point = np.array([0,0,1,0])
typeLabels = ["Smoker", "Obese", "Exercise", "Smoke&Obese", "Obest&Exercise", "Cure"]
testCases = [np.array([1,0,0,0]), np.array([0,1,0,0]), np.array([0,0,1,0]), np.array([1,1,0,0]), np.array([0,1,1,0]), np.array([0,0,0,1])]

caseLoop = 0
for case in testCases:
    result = sigmoid(np.dot(case, weights) + bias)
    resultPercent = result*100
    print(typeLabels[caseLoop] + ": %.2f" % resultPercent + "%")
    caseLoop = caseLoop+1