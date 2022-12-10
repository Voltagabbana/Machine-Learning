import sys
import pandas as pd

inputs = sys.argv
assigner = lambda name: inputs[inputs.index("--"+name)+1]
path = assigner("data")
eta = assigner("eta")
threshold = assigner("threshold")

import numpy as np
dataset = np.genfromtxt(path,delimiter=",")
xt1 =dataset[0:,0]
xt2 = dataset[0:,1]
yt = dataset[0:,2]

w0 = 0
w1 = 0
w2 = 0
eta = 0.00005
threshold = 0.0001
it = 0
diff = threshold*2
seaft = 10000000000

while (diff) > threshold:
  sebef = seaft
  seaft = 0
  grad0 = 0
  grad1 = 0
  grad2 = 0
  for i in range (0,1000):
    err = yt[i]-(w0+w1*xt1[i]+w2*xt2[i]) # == y_i - f(x_i)
    grad0 = 1*err+grad0
    grad1 = xt1[i]*err + grad1
    grad2 = xt2[i]*err + grad2
    seaft = seaft+(err)**2
  print(it,w0,w1,w2,seaft)
  w0 = w0 + eta*grad0
  w1 = w1 + eta*grad1
  w2 = w2 + eta*grad2
  diff = sebef-seaft
  it += 1


    
