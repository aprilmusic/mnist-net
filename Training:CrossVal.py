import numpy as np
from numpy import array
import random
import os
import _pickle as pickle

with open(r"train-images-idx3-ubyte", "rb") as f: #automatically open, close file
    data = f.read()

with open(r"train-labels.idx1-ubyte", "rb") as g: #automatically open, close file
    dataLabels = g.read()

with open(r"t10k-images.idx3-ubyte", "rb") as h1: #automatically open, close file
    dataTest = h1.read()
    
with open(r"t10k-labels.idx1-ubyte", "rb") as h:
    dataLabelsTest = h.read()

numberDigits = 10000
epochs = 100
inputLayerSize = 28**2
hiddenLayerSizes = [100,100]
outputLayerSize = 10 #each digit
#lambda
learnrates = [0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03]

#convert lists to arrays
#np.asarray(inputs)
#np.asarray(correct)

#randomly generate starting weights. There are weights between every two layers.
weights = []
weights.append(np.random.uniform(low = -.2, high = .2, size=(inputLayerSize,hiddenLayerSizes[0])))
for i in range(len(hiddenLayerSizes)-1):
    weights.append(np.random.uniform(low = -.2, high =.2, size=(hiddenLayerSizes[i],hiddenLayerSizes[i+1])))
weights.append(np.random.uniform(low = -.2, high = .2, size=(hiddenLayerSizes[-1],outputLayerSize)))


#randomly generate starting biases. There is bias at every layer except input.
biases = []
for i in range(len(hiddenLayerSizes)):
    biases.append(np.random.uniform(low = -.1, high =.1, size=(1,hiddenLayerSizes[i])))
biases.append(np.random.uniform(low = -.1, high = .1, size=(1,outputLayerSize)))

def train(x,c,learnrate):
    '''x is array of inputs and c is correct output value.
    '''
    global weights, biases
    activations = [array(x).T]
    input_sums = [array(x)]
    #Now run feedforward
    for weight,bias in zip(weights,biases): #because same length and zip is cool!
        input_sums.append(activations[-1].dot(weight)+bias)
        activations.append(sigmoid(input_sums[-1])) #this line does all the work
    result = np.argmax(activations[-1])
    if result != c: #if didn't choose the right node
        deltaSigs = []
        deltaSigs.append(np.copy(input_sums[-1])) #going backwards, E = Z-c
        #need to use copy in order to change one specific list in list
        deltaSigs[0][0][c] -= 1
        #Now adjust weights and biases using the delta signal
        for i in range(len(weights)):
            deltaSig = deltaSigs[-1].dot(weights[-i-1].T)
            deltaSig1 = deltaSig*sigmoid_prime(activations[-i-2])
            deltaSigs.append(deltaSig1) #creating delta signals
        for i in range(len(weights)):
            weights[i] -= learnrate * activations[i].T.dot(deltaSigs[-i-2]) #don't want to include the output activation, and using last set of delta signals
            #when deltaSig (which is like error for a certain node) is large, then weight goes down so it factors in less
            biases[i] -= deltaSigs[-i-2] * learnrate #is like weights, where activation is 1 :D
        #print(result,c)
    #elif result == c:
        #print("yay!", result, c)

def sigmoid(x): #errors will always be between 0 and 1
    '''The activation function.'''
    return 1/(1+np.exp(-x))

def sigmoid_prime(y): #looks nice with output which we store :D <3 yaaay
    '''Derivative of the activation function. Used in backpropagation.'''
    return y*(1-y)
'''
for i in range(epochs):
    for j in range(numberDigits):  #inputs should be an array of numberDigits 784*1 arrays
        train(inputs[j],correct[j])'''

#generate inputs, training
for index, learnrate in enumerate(learnrates):
    for i in range(epochs):
        for j in range(numberDigits):
            label = dataLabels[j+8]
            digit = []
            for k in range(16+784*j, 784*j + 800):
                digit.append(data[k])
            digit = array(digit)
            digit = digit.reshape(digit.shape[0],-1)
            #inputs.append(digit.T)
            
            train(digit, label, learnrate)

        if i % 100 == 0:
            pickle.dump(weights, open("weightsc-%d.p" % index, "wb"))
            pickle.dump(biases, open("biasesc-%d.p" % index, "wb"))
            print("Saving at i =", i)
            

accuracies = []
#check on cross-validation set
for index, learnrate in enumerate(learnrates):
    weights = pickle.load(open(r"weightsc-%d.p" % index, "rb"))
    biases = pickle.load(open(r"biasesc-%d.p" %index, "rb"))

    numTest = 1000
    correctTest = 0

    for i in range(numTest):
        label = dataLabelsTest[i+8]
        digit = []
        for j in range(16+784*i, 16+784*i +784):
            digit.append(dataTest[j])
        digit = array(digit)
        digit = digit.reshape(digit.shape[0],-1)
        #inputs.append(digit.T)

        activations2 = [array(digit.T)]
        for weight,bias in zip(weights,biases): #because same length and zip is cool! 
            activations2.append(sigmoid(np.dot(activations2[-1],weight)+bias)) #this line does all the work
        result = np.argmax(activations2[-1]) #result from neural net
        if label == result:
            correctTest += 1
        #print(label)
        #print(result)

    accuracies.append(correctTest / numTest)

for i in range(len(learnrates)):
    print(str(learnrates[i]) + ": " + str(accuracies[i]))


os.system("say 'beeeeeeep'")
