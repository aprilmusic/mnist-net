from statistics import mean
import scipy
from scipy import ndimage
from scipy import misc
import PIL
import numpy as np
from PIL import Image
import random

'''
#convert things to grayscale
im = Image.open("wall9.jpg").convert('LA')
im.show()


#convert images to arrays
image = Image.open("wall9.jpg") #open image 
misc.imsave("wall9.jpg",image) #save it so misc can act on it
image = misc.imread("wall9.jpg") #reads it to an array
print(type(image))


#convert array to image
wall2 = Image.fromarray(wall)
wall2.save("wall9.jpg")
print(type(wall2))
'''


def MNISTdigits(i):
    '''Threshold between 0 and 255, number of images less than 60000. Prints them in form of hashtag.'''
    with open(r"train-images-idx3-ubyte", "rb") as f: 
        data = f.read()
    print(data[i])

       
def inputs():
    digit = input("What is your favorite digit?")
    threshold = input("Enter a digit between 0 and 255: ")

def prettydigits(digit,number):
    '''Threshold between 0 and 255, number of images less than 60000. Prints them in form of hashtag.'''
    with open(r"training.pix", "rb") as f: 
        data = f.read()
    with open(r"training_labels.txt", "rb") as g: 
        dataLabels = g.read()
    digits = []
    for i in range(1000):
        if dataLabels[i] == digit:
            digits.append(i)
    random.shuffle(digits)
    for n in range(number): 
        image = []
        for i in range(15):
            image.append([])
            for j in range(15):
                if data[225*(digits[n]) + 15 * i + j] > 3:
                    image[i].append(" ")
                elif data[225*(digits[n]) + 15 * i + j] > 2.5:
                    image[i].append(".")
 #               elif data[225*(digits[n]) + 15 * i + j] < 3:
 #                   image[i].append("X")
                else:
                    image[i].append("#")
        for i in range(15): #prints image, line by line
            print ("".join(image[i]))

prettydigits(5,15)
    

def averageMathcamper(digit):
    with open(r"training.pix", "rb") as f: 
        data = f.read()
    with open(r"training_labels.txt", "rb") as g: 
        dataLabels = g.read()
    digits = []
    for i in range(1000):
        if dataLabels[i] == digit:
            digits.append(i)
    digitsdata = []
    for j in range(len(digits)):
        singledigit = []
        for k in range(225*digits[j], 225*digits[j] + 225):
            singledigit.append(data[k])
        digitsdata.append(singledigit)
    digitsdata = np.asarray(digitsdata)
    average = np.mean(digitsdata,axis = 0)
    image = []
    for i in range(15):
        image.append([])
        for j in range(15):
            if average[15 * i + j] > 2.4:
                image[i].append(" ")
 #           elif average[15 * i + j] > 2.7:
 #               image[i].append(".")
 #          elif average[15 * i + j] < 2.5:
#               image[i].append("X")
            else:
                image[i].append("#")
    print("The average Mathcamper " + str(digit) + " looks like this:")
    for i in range(15): #prints image, line by line
        print ("".join(image[i]))
    print("This mess is YOUR fault.")

def newpage():
    for i in range(30):
        print('\n')




















#Enter your favorite digit and how many times you would like to see it,
#in Mathcamp's own handwriting!
        
digit = 5
num_of_digits = 8 #less than 15 please!


#press fn and f5, then click OK to run!

#These digits came from YOU!



#prettydigits(digit, num_of_digits)













































