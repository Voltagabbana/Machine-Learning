import sys
import math
import pandas as pd
import numpy as np

##### Importo i dati i creo i vettori colonna necessari ecc.####

# INPUT (codice di emin)
inputs = sys.argv
assigner = lambda name: inputs[inputs.index("--"+name)+1]
path = assigner("data")
eta = assigner("eta")
threshold = assigner("iterations")

# USO PANDAS

#path = "Gauss4.csv"
data = pd.read_csv(path,header=None)
data.columns = ['a','b','t']
#print(data)

a =data.iloc[:,0] # qua ho un dataframe che seleziona solo una colonna
avec = a.values[:] # qua invece metto quella colonna in un array  

#print(avec)

b =data.iloc[:,1] # qua ho un dataframe che seleziona solo una colonna
bvec = b.values[:] # qua invece metto quella colonna in un array  #['high' 'low' 'med' 'vhigh']

t =data.iloc[:,2] # qua ho un dataframe che seleziona solo una colonna
tvec = t.values[:] # qua invece metto quella colonna in un array  #['high' 'low' 'med' 'vhigh']

datlen = len(avec) #6000

nu = float(eta) # devo dire io che tipo di input è, altrimenti pensa sia una stringa!

#print(nu)
#print(type(nu)) # devo dire io che tipo di input è, altrimenti pensa sia una stringa!


w_bias_h1 = 0.2
w_a_h1 = -0.3
w_b_h1 = 0.4
w_bias_h2 = -0.5
w_a_h2 = -0.1
w_b_h2 = -0.4
w_bias_h3 = 0.3
w_a_h3 = 0.2
w_b_h3 = 0.1
w_bias_o = -0.1
w_h1_o = 0.1
w_h2_o = 0.3
w_h3_o = -0.4
nit = int(threshold) # # devo dire io che tipo di input è, altrimenti pensa sia una stringa!
#print(nit)
#print(type(nit)) # devo dire io che tipo di input è, altrimenti pensa sia una stringa!
print('-','-','-','-','-','-','-','-','-','-','-',w_bias_h1,
            w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o) #provo senza "\n"
for it in range(0,nit):
    for i in range (0,datlen):
            # calcolo gli output degli hidden nodes e dell'output node
            in_h1 = w_bias_h1*1+avec[i]*w_a_h1+bvec[i]*w_b_h1
            #print(in_h1)
            out_h1 = 1/(1+math.exp(-in_h1))
            #print(out_h1)

            in_h2 = w_bias_h2*1+avec[i]*w_a_h2+bvec[i]*w_b_h2
            #print(in_h2)
            out_h2 = 1/(1+math.exp(-in_h2))
            #print(out_h2)

            in_h3 = w_bias_h3*1+avec[i]*w_a_h3+bvec[i]*w_b_h3
            #print(in_h3)
            out_h3 = 1/(1+math.exp(-in_h3))
            #print(out_h3)

            in_o = w_bias_o*1+out_h1*w_h1_o+out_h2*w_h2_o+out_h3*w_h3_o
            #print(in_o)
            out_o = 1/(1+math.exp(-in_o))
            #print(out_o) 

            # Ok, per il primo dato i risultati sono giusti quindi idea corretta!
            

            # calcolo i delta
            delta_o = out_o*(1-out_o)*(tvec[i]-out_o)
            delta_h1 = out_h1*(1-out_h1)*delta_o*w_h1_o
            delta_h2 = out_h2*(1-out_h2)*delta_o*w_h2_o
            delta_h3 = out_h3*(1-out_h3)*delta_o*w_h3_o


            # aggiorno i pesi

            deltaw_bias_h1 = nu*delta_h1
            deltaw_a_h1 = nu*delta_h1*a[i]
            deltaw_b_h1 = nu*delta_h1*b[i]
            deltaw_bias_h2 = nu*delta_h2
            deltaw_a_h2 = nu*delta_h2*a[i]
            deltaw_b_h2 = nu*delta_h2*b[i]
            deltaw_bias_h3 = nu*delta_h3
            deltaw_a_h3 = nu*delta_h3*a[i]
            deltaw_b_h3 = nu*delta_h3*b[i]
            deltaw_bias_o = nu*delta_o
            deltaw_h1_o = nu*delta_o*out_h1
            deltaw_h2_o = nu*delta_o*out_h2
            deltaw_h3_o = nu*delta_o*out_h3

            w_bias_h1 = w_bias_h1 + deltaw_bias_h1
            w_a_h1 = w_a_h1 + deltaw_a_h1
            w_b_h1 = w_b_h1 + deltaw_b_h1
            w_bias_h2 = w_bias_h2 + deltaw_bias_h2
            w_a_h2 = w_a_h2 + deltaw_a_h2
            w_b_h2 = w_b_h2 + deltaw_b_h2
            w_bias_h3 = w_bias_h3 + deltaw_bias_h3
            w_a_h3 = w_a_h3 + deltaw_a_h3
            w_b_h3 = w_b_h3 + deltaw_b_h3
            w_bias_o = w_bias_o + deltaw_bias_o
            w_h1_o = w_h1_o + deltaw_h1_o
            w_h2_o = w_h2_o + deltaw_h2_o
            w_h3_o = w_h3_o + deltaw_h3_o
            print(avec[i],bvec[i],out_h1,out_h2,out_h3,out_o,tvec[i],delta_h1,delta_h2,delta_h3,delta_o,w_bias_h1,
            w_a_h1,w_b_h1,w_bias_h2,w_a_h2,w_b_h2,w_bias_h3,w_a_h3,w_b_h3,w_bias_o,w_h1_o,w_h2_o,w_h3_o)