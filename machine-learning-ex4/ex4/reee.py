import numpy as np
import scipy.io
from scipy import optimize



def randInitializeWeights(a,b,ep):
    
    return np.random.rand(a,b)*(2*ep)-ep

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predictOneVsAll(theta1,theta2, X):
    (m,n)=X.shape
    a2=sigmoid(theta1*X.transpose())
    a2=np.matrix(np.concatenate(( np.ones((1,a2.shape[1])) ,a2), axis=0) ) #add the bias
    H=np.array(sigmoid(theta2*a2)).transpose()
    p= [np.argmax(H[i])+1 for i in range(m)]

    return p

input_layer_size  = 400
hidden_layer_size = 25 
num_labels =N_labels= 10
m=5000
Data=scipy.io.loadmat('ex4data1.mat')
X,y=Data['X'],Data['y'].flatten()
X=np.matrix(np.concatenate(( np.ones((m,1)) ,X), axis=1) )
(m,n)=X.shape
wights=scipy.io.loadmat('ex4weights.mat')
theta=np.array(np.concatenate((  wights['Theta1'].flatten() ,wights['Theta2'].flatten() ) ))



lambda_=1

theta1=wights['Theta1']
theta2=wights['Theta2']#+1 for thr hidden unit
a2=sigmoid(theta1*X.transpose())
a2=np.matrix(np.concatenate(( np.ones((1,a2.shape[1])) ,a2), axis=0) ) #add the bias
a3=sigmoid(theta2*a2)
unrolled_y=np.array( [np.array(y==i).astype(int) for i in range(1,a3.shape[0]+1) ] ) # 10 is supposed ot be the last on the list when u replaced 10 with 0 it became the first 
a3=np.array(a3) #need array multi insted of matrix multy so i get 1 on 1 multy 

S_delta3=np.matrix(a3-unrolled_y)
a2=np.array(a2)
S_delta2=np.array(np.matrix(theta2).transpose()*S_delta3) *  (a2*(1+a2.dot(-1))) 
    
    
Del2=S_delta3*a2.transpose()
Theta2_grad = np.zeros(theta2.shape);
for i in range(theta2.shape[0]):
        for j in range(theta2.shape[1]):
            Theta2_grad[i,j]=Del2[i,j]/m
            if j!=0:
                Theta2_grad[i,j]+=(lambda_/m)*theta2[i,j]
                
theta1[:,1]=0
theta2[:,1]=0
grad_reg_term=lambda_*np.array( [ theta2, theta1 ] )
DELTA = (1/m)*(np.array( [ S_delta3*a2.transpose()  , S_delta2[1:,:]*X ]  ) + grad_reg_term) # S_delta2[1:,:] to ignore the bias?
    
    
    
print(DELTA[1][0] ) # this is wrong check the grad_reg for the second one.. u ched everything else like twice already maybe thess arent allingened or smt and the first one is the only correct one prob cuz no correcting 0 so maybe u can use that to figure out ifit's an allingment issu 

