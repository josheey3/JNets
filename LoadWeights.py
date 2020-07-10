import numpy as np
import pandas

def sigmoid(x):
    return 1/(1+np.exp(-x))

weights = np.array([np.array([-0.02698402]), np.array([6.07872098]), np.array([-0.64030568])])
bias = np.array([-2.4607081638368067])

while(True): 
    print("--------")
    inSmoker = float(input("Do you smoke? "))
    inObese = float(input("Do are you obese? "))
    inExercise = float(input("Do you exercise? "))

    single_point = np.array([inSmoker,inObese,inExercise])
    result = sigmoid(np.dot(single_point, weights) + bias)
    resultPercent = result*100
    print("Chance of diabities: %.2f" % resultPercent + "%")