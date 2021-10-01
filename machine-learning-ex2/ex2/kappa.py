import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

def plotdata(f1,f2,line=False,killme=False):
    x1,y1,x2,y2=[],[],[],[]
    for i in range(m):
        if y[i]:x1.append(f1[i]),y1.append(f2[i])
        else:x2.append(f1[i]),y2.append(f2[i])
    plt.plot(x1, y1, 'rx')
    plt.plot(x2, y2, 'bo')
    plt.ylabel('exame 1')
    plt.xlabel('exame 2')
    plt.xticks(np.arange(min(f1), max(f1)+1, 50))
    plt.yticks(np.arange(min(f2), max(f2)+1, 50))
    plt.legend(['ex1','ex2'])
    if line:
        l1 = np.array([min(f1),max(f1)])
        l2=(-1./theta[2])*(theta[1]*l1 + theta[0])
        plt.plot(l1, l2, '-g')
    if killme: # haha fix this 
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z=np.zeros((len(u),len(v)))
        for i in range(1,len(u)+1):
            for j in range(1,len(v)+1):
                z[i,j]=mapFeature(u[i],v[j])*theta 
        z=z.transpose()
        plt.contour(u, v, z, [0, 0], 'LineWidth', 2)   
    plt.show()
    
    
def sigmoid(z):
    return 1/(1+np.exp(-z))
def costFunction(theta, X, y):
    H=np.array(sigmoid(theta*X.transpose()))
    cost=(-1/m)*np.sum( y*np.log(H) + (1-y)*np.log(1-H)  )
    grad=(1/m)*(H-y)*X
    return cost,grad
def predict(x):
    pred=sigmoid(theta*np.matrix(x).transpose()).tolist()[0]
    pred=[1 if i>=.5 else 0 for i in pred]
    return pred
def mapFeature(X1,X2):
    degree = 6
    out=np.matrix(np.ones((m,1)))
    for i in range(1,degree+1):
        for j in range(i+1):
            out=np.concatenate( ( out,  np.multiply( np.power(X1,i-j),np.power(X2,j) ) ) ,axis=1)
    return out         
            
def costFunctionReg(theta, X, y, lambd):
    H=np.array(sigmoid(theta*X.transpose()))
    j=costFunction(theta, X, y)[0]+(lambd/(2*m))*np.sum(np.square(theta)[1:])
    reg=(lambd/m)*theta
    reg[0]=0
    grad=(1/m)*(H-y)*X+reg
    return j,grad
    

data = open('ex2data2.txt', 'r').read().replace('\n',',').split(',')       # 2 ,118  for multi and 1 ,100
data =np.matrix( [float(i) for i in data]).reshape((118,3))
y=np.array(data[:,-1]).flatten()
X=data[:,:-1]
(m,n)=X.shape
#X=np.concatenate(( np.ones((m,1)) ,X), axis=1)                           #one or the other 
X=mapFeature(X[:,0] ,X[:,1])                                            # one or the other  for multi 
(m,n)=X.shape
#plotdata(X[:,1].flatten().tolist()[0],X[:,2].flatten().tolist()[0])
theta=np.zeros(n)
lambd = 1

cost,grad=costFunctionReg(theta, X, y, lambd)

options= {'maxiter': 400}




#opt=scipy.optimize.minimize(costFunction,theta,(X,y),method='TNC',jac=True,options=options)#one or the other
opt=scipy.optimize.minimize(costFunctionReg,theta,(X,y,lambd),method='TNC',jac=True,options=options)# one or the other  for multi 
cost=opt.fun
theta = opt.x
p = predict(X)
print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')

    

#plotdata(X[:,1].flatten().tolist()[0],X[:,2].flatten().tolist()[0],True)






