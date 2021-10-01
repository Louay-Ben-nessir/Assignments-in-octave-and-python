import numpy as np
import scipy.io
from scipy import optimize

def lrCostFunction(theta,X,y,lambd):
    if y.dtype == bool:y = y.astype(int) # this single line cost me like 4 h 
    (m,n)=X.shape
    H=np.array(sigmoid(theta*X.transpose()))
    j= (1/m) * np.sum((-y*np.log(H)-(1-y)*np.log(1-H))) + (lambd/(2*m))* np.sum(np.square(theta[1:]))
    theta_nz=np.concatenate(([0],theta[1:]))
    grad=np.matrix( (1/m)*(H-y)*X+(lambd/m)*theta_nz )
    print(type(grad))
    return j,grad

def oneVsAll(X,y,lambd,labels):
    options = {'maxiter': 50}
    theta_all=np.zeros((labels,input_n)) 
    theta=np.zeros(input_n)
    for i in range(labels):
        res = optimize.minimize(lrCostFunction, 
                                theta, 
                                (X, (y == i), lambd), 
                                jac=True, 
                                method='TNC',
                                options=options) 
        theta_all[i,:]=res.x
    return theta_all


def predictOneVsAll(theta, X):
    (m,n)=X.shape
    H=np.array(sigmoid(X*theta.transpose()))
    p= [np.argmax(H[i]) for i in range(m)]
    return p

def predict(theta, X):
    (m,n)=X.shape
    H=np.array(sigmoid(X*theta.transpose()))
    p= np.argmax(H)
    return p

def sigmoid(z):
    return 1/(1+np.exp(-z))

#init====================================================
data = scipy.io.loadmat('ex3data1.mat')
X,y=data['X'],data['y'].flatten()
(m,input_n)=X.shape
X=np.matrix(np.concatenate(( np.ones((m,1)) ,X), axis=1) )
(m,input_n)=X.shape
y[y == 10] = 0 #aha totally didnt steal this looooool
y_vals=10
#================loading and visulazing Data=============
#==============Vectorize Logistic Regression============= 
theta_t = np.array([-2, -1, 1, 2], dtype=float)
X_t = np.matrix(np.concatenate([np.ones((5, 1)), np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)) #didnt steal this one either 
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3
l=lrCostFunction(theta_t,X_t,y_t,lambda_t)
lambd = 0.1
theta_F=oneVsAll(X,y,lambd,y_vals)



p=predictOneVsAll(theta_F, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(p == y) * 100))
#ez clapssssssssssssssssssssssssssssss haha ;-; no nn cuz only predict And i kinda wanna do all of the nn alone
