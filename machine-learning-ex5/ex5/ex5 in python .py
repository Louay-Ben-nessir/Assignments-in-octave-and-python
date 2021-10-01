import numpy as np
import scipy.io
from scipy import optimize
import matplotlib.pyplot as plt


def linearRegCostFunction(theta,X,y,lambda_):
    (m,n)=X.shape
    theta=np.matrix(theta)
    H=X*theta.transpose()
    theta[:,0]=0
    j=( 1/(2*m) )* ( np.sum(np.square(H-y) ) + lambda_*np.sum( np.square(theta) ) )
    grad=(1/m)*(  (X.transpose()*np.matrix(H-y)).transpose() +  np.matrix( lambda_* theta) )  
    return j,grad
    
def learningCurve(X,y,X_cv,y_cv,lambda_):
    (m,n)=X.shape
    error_train = np.zeros((m, 1))
    error_val   = np.zeros((m, 1))
    theta       = np.zeros((1,n))
    for i in range(2,m):
        res = optimize.minimize(linearRegCostFunction, 
                                theta, 
                                (X[:i,:],y[:i],lambda_), 
                                jac=True, 
                                method='TNC',
                                options={'maxiter': 100})
        error_train[i,:],temp=linearRegCostFunction(res.x,X[:i,:], y[:i], lambda_)
        error_val[i,:],temp=linearRegCostFunction(res.x ,X_cv, y_cv, lambda_)
    return error_train,error_val

def poly(X,p):
    X_poly = zeros(X.shape[0], p)
    for i in range(1,p+1):X_poly[:,i]=np.power(X[:,1],i)
    return X_poly
        
def validationCurve():
    pass

Data=scipy.io.loadmat('ex5data1.mat')


plt.plot(Data['X'],Data['y'], 'rx')
plt.ylabel('Water flowing out of the dam (y)')
plt.xlabel('Change in water level (x)')
plt.xticks(np.arange(min(Data['X']), max(Data['X'])+1, 50))
plt.yticks(np.arange(min(Data['y']), max(Data['y'])+1, 50))

theta=np.matrix([1,1])
lambda_=1

X=np.concatenate(( np.ones((Data['X'].shape[0],1)) ,Data['X']), axis=1) 
X_cv=np.concatenate(( np.ones((Data['Xval'].shape[0],1)) ,Data['Xval']), axis=1) 
cost,grad=linearRegCostFunction( theta,X,np.array(Data['y']),lambda_)


plt.plot(Data['X'], X*theta.transpose(), '-g',)

plt.show()


error_train,error_val=learningCurve(X,Data['y'],X_cv,Data['yval'],lambda_)

plt.plot(range(2,m),error_train[2:], '-b')
plt.plot(range(2,m),error_val[2:], '-g')
plt.ylabel('Error')
plt.xlabel('Number of training examples')
plt.legend(['training','Cross Validation'])
plt.show()

