import numpy as np
import copy

inputs=[[0,0,0,0,0,0],[1,0,1,0,0,0],[1,0,0,1,1,1],[0,1,0,1,0,1],[0,1,1,0,1,0]]

N = [len(k) for k in inputs][0]
P=len(inputs)
#Create empty weight matrix
w = np.zeros((N, N))
#To store learning rule 1 vs. -1
inputs_weight=[]
#Convert inputs to an array
inputs=np.array(inputs)

### Apply learning rule ### 

#Change input to either 1 or -1 according to learning rule
for i in inputs:
    for j in i:
        if j < 1 :
            inputs_weight=np.append(inputs_weight,-1)
        else:
            inputs_weight=np.append(inputs_weight,1)
    
inputs_weight= np.array_split(inputs_weight, len(inputs))
inputs_weight=np.array(inputs_weight)

#Weight matrix
for i in range(N):
    for j in range(N):
        for p in range(P):
            w[i, j] += (inputs_weight[p, i]*inputs_weight[p, j])
        if i==j:
            w[i, j] = 0

### Iteration and apply performance rule ###
for iterations in range (10):
    outputs= []
    #s matrix
    for input in inputs:
        s = np.zeros((len(inputs[0],)))
        for j in range(np.shape(w)[0]):
            for i in range(np.shape(w)[1]):
                s[j] += input[i] *w[i][j]
        #Apply s condition    
        for k in range(len(s)):
            if s[k]>0:
                s[k]=1
            elif s[k]<0:
                s[k] = 0
            else:
                s[k]=input[k]      
        outputs.append(s)
    #update the inputs for next iteration with the previous iteration output
    inputs = copy.deepcopy(outputs) 
    
print(outputs) 
