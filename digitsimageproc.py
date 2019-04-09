import numpy as np
from numpy import array
import random
import _pickle as pickle
from PIL import Image

weights = pickle.load(open(r"weights-%d.p" % 1700, "rb"))
biases = pickle.load(open(r"biases-%d.p" % 1700, "rb"))

def sigmoid(x): #errors will always be between 0 and 1
    '''The activation function.'''
    return 1/(1+np.exp(-x))

THRESHOLDS = [190,210,245,255]
#THRESHOLDS = [255-(255-x)//3 for x in THRESHOLDS]

def get_val(x):
  for i,t in enumerate(THRESHOLDS):
    if x <= t:
      return i

CHARS = "#X. "

DIGITS = 10
INPUT = "input3.png"

BORDER = 1
WIDTH = 168
HEIGHT = 168
PIXEL_SIZE = 6

im = Image.open(INPUT)
g = im.convert("L")

pixels = []
images = []
for outery in range(2):
  for outerx in range(5):
    X_OFFSET = BORDER*(2+4*outerx) + WIDTH*outerx
    Y_OFFSET = BORDER*(2+4*outery) + HEIGHT*outery
    new_image = []
    for y in range(WIDTH//PIXEL_SIZE):
      new_image.append([])
      for x in range(HEIGHT//PIXEL_SIZE):
        new_image[-1].append(3-get_val(sum(g.getpixel((X_OFFSET+x*PIXEL_SIZE+subx, Y_OFFSET+y*PIXEL_SIZE+suby)) for subx in range(PIXEL_SIZE) for suby in range(PIXEL_SIZE))//(PIXEL_SIZE**2)))
        pixels.append(new_image[-1][-1])
    images.append("\n".join("".join(CHARS[3-x] for x in line) for line in new_image))

results = []
for i in range(DIGITS):
  digit = []
  for j in range(784*i, 784*i +784):
      digit.append(pixels[j])
  digit = array(digit)
  digit = digit.reshape(digit.shape[0],-1)
  #inputs.append(digit.T)

  activations2 = [array(digit.T)]
  for weight,bias in zip(weights,biases): #because same length and zip is cool! 
      activations2.append(sigmoid(np.dot(activations2[-1],weight)+bias)) #this line does all the work
  result = np.argmax(activations2[-1]) #result from neural net
  rankings = sorted([(r,i) for i,r in enumerate(activations2[-1][0])],reverse=True)
  results.append("Guesses: %s, %s, %s, %s"%(rankings[0][1], rankings[1][1], rankings[2][1], rankings[3][1]))

for i,r in zip(images,results):
  print(i)
  print(r)
  print("---------------------------------------------")
