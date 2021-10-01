import numpy as np
import scipy.io
from scipy import optimize

def nnCostFunction(theta, input_layer_size, hidden_layer_size, N_labels, X, y, lambda_):
    (m,n)=X.shape
    theta1=theta[0:(n*hidden_layer_size)].reshape((hidden_layer_size,n))
    theta2=theta[(n*hidden_layer_size):].reshape((N_labels,hidden_layer_size+1))#+1 for thr hidden unit
    a2=sigmoid(theta1*X.transpose())
    a2=np.matrix(np.concatenate(( np.ones((1,a2.shape[1])) ,a2), axis=0) ) #add the bias
    a3=sigmoid(theta2*a2)
    unrolled_y=np.array( [np.array(y==i).astype(int) for i in range(1,a3.shape[0]+1) ] ) # 10 is supposed ot be the last on the list when u replaced 10 with 0 it became the first 
    a3=np.array(a3) #need array multi insted of matrix multy so i get 1 on 1 multy 
    base=(1/m)*np.sum ( ( unrolled_y.dot(-1)*np.log(a3) - ( 1 + unrolled_y.dot(-1) )*np.log( 1+a3.dot(-1) ) ) )
    reg_term= (lambda_/(2*m))*(np.sum(np.square(theta1[:,1:]))+np.sum(np.square(theta2[:,1:])))
    cost=base+reg_term
    
    S_delta3=np.matrix(a3-unrolled_y)
    a2=np.array(a2)
    S_delta2=np.array(np.matrix(theta2).transpose()*S_delta3) *  (a2*(1+a2.dot(-1)))
    
     '''Del2=S_delta3*a2.transpose()
    Del1=S_delta2[1:,:]*X
    
   Theta1_grad = np.zeros(theta1.shape)
    for i in range(theta1.shape[0]):
        for j in range(theta1.shape[1]):
            Theta1_grad[i,j]=Del1[i,j]/m
            if (j !=1):
                Theta1_grad[i,j]+=(lambda_/m)*theta1[i,j]
        
    Theta2_grad = np.zeros(theta2.shape)
    for i in range(theta2.shape[0]):
        for j in range(theta2.shape[1]):
            Theta2_grad[i,j]=Del2[i,j]/m
            if (j !=1):
                Theta2_grad[i,j]+=(lambda_/m)*theta2[i,j]
            
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])'''# weak ppl implelmntation 
    
    S_delta3=np.matrix(a3-unrolled_y)
    a2=np.array(a2)
    S_delta2=np.array(np.matrix(theta2).transpose()*S_delta3) *  (a2*(1+a2.dot(-1))) 
                
    theta1[:,1]=0
    theta2[:,1]=0
    grad_reg_term=lambda_*np.array( [ theta2, theta1 ] )
    DELTA = (1/m)*(np.array( [ S_delta3*a2.transpose()  , S_delta2[1:,:]*X ]  ) + grad_reg_term) # S_delta2[1:,:] to ignore the bias?
    
    DELTA=np.concatenate( ( np.array(DELTA[1]).ravel() ,np.array(DELTA[0]).ravel() ) ) 
   
  
    return cost,DELTA

def randInitializeWeights(a,b,ep):
    return np.random.rand(a,b)*2*ep-ep

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
num_labels = 10
m=5000
Data=scipy.io.loadmat('ex4data1.mat')
X,y=Data['X'],Data['y'].flatten()
X=np.matrix(np.concatenate(( np.ones((m,1)) ,X), axis=1) )
(m,n)=X.shape
wights=scipy.io.loadmat('ex4weights.mat')
theta_VEC=np.array(np.concatenate((  wights['Theta1'].flatten() ,wights['Theta2'].flatten() ) ))

lambda_ = 1
cost,grad=nnCostFunction(theta_VEC, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)
ep=0.12
int_theta1=randInitializeWeights(hidden_layer_size,n,ep)
int_theta2=randInitializeWeights(num_labels,hidden_layer_size+1,ep)
int_theta_VEC=np.array(np.concatenate((  int_theta1.flatten() ,int_theta2.flatten() ) ))


p=predictOneVsAll(int_theta1,int_theta2, X)
print('Training Set Accuracy befor training:  {:.2f}%'.format(np.mean(p == y) * 100))


options = {'maxiter': 100}
res = optimize.minimize(nnCostFunction, 
                                int_theta_VEC, 
                                (input_layer_size, hidden_layer_size, num_labels, X, y, lambda_), 
                                jac=True, 
                                method='TNC',
                                options=options) 

theta1=res.x[0:(n*hidden_layer_size)].reshape((hidden_layer_size,n))
theta2=res.x[(n*hidden_layer_size):].reshape((num_labels,hidden_layer_size+1))
p=predictOneVsAll(theta1,theta2, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(p == y) * 100))

