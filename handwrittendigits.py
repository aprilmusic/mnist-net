import numpy as np
from numpy import array
import random
import _pickle as pickle


with open(r"C:\Users\jsu\Desktop\training.pix", "rb") as f: #automatically open, close file
    train_data = f.read()

with open(r"C:\Users\jsu\Desktop\training_labels.txt", "rb") as g: #automatically open, close file
    train_labels = g.read()

with open(r"C:\Users\jsu\Desktop\testing.pix", "rb") as h1: #automatically open, close file
    test_data = h1.read()
    
with open(r"C:\Users\jsu\Desktop\testing_labels.txt", "rb") as h:
    test_labels = h.read()

numberDigits = 1012
epochs = 100000000
inputLayerSize = 15**2
hiddenLayerSizes = [40,40]
outputLayerSize = 10 #each digit
#lambda
learnrate = 0.07

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


def train(x,c):
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


#generate inputs, training
for i in range(epochs):
    for j in range(numberDigits):
        label = train_labels[j]
        
        digit = []
        for k in range(225*j, 225*j + 225):
            digit.append(train_data[k])
        digit = array(digit)
        digit = digit.reshape(digit.shape[0],-1)
        #inputs.append(digit.T)
        
        train(digit, label)

    if i % 100 == 0:
        pickle.dump(weights, open("weights-%d.p" % i, "wb"))
        pickle.dump(biases, open("biases-%d.p" % i, "wb"))
        print("Saving at i =", i)

print("Done!")
#print(weights)
#print(biases)
    
#testing data
numTest = 500
correctTest = 0


for i in range(numTest):
    label = test_labels[i]
    
    digit = []
    for j in range(225*i, 225*i +225):
        digit.append(test_data[j])
    digit = array(digit)
    digit = digit.reshape(digit.shape[0],-1)
    #inputs.append(digit.T)

    activations2 = [array(digit.T)]
    for weight,bias in zip(weights,biases): #because same length and zip is cool! 
        activations2.append(sigmoid(np.dot(activations2[-1],weight)+bias)) #this line does all the work
    result = np.argmax(activations2[-1]) #result from neural net
    if label == result:
        correctTest += 1

print(correctTest / numTest)
