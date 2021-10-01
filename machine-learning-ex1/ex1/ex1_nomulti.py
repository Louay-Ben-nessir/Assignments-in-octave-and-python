import matplotlib.pyplot as plt
import numpy as np

def plotdata(x,y,line=False):
    plt.plot(x, y, 'rx')
    plt.ylabel('h(heta) or y ')
    plt.xlabel('feature x')
    plt.xticks(np.arange(min(x), max(x)+1, 2))
    plt.yticks(np.arange(min(y), max(y)+1, 2))
    if line:
        l1 = np.linspace(min(x),max(x))
        l2 = THETA[1]*l1+THETA[0]
        plt.plot(l1, l2, '-b', label='y=2x+1')
    plt.show()
    
def gradientDescent(X, y, theta, alpha, iterations):#if ever confused do the math again and it will make sense
    for i in range(iterations):
        H=X.dot(theta)
        l=np.array(H-y)
        r=l*X
        theta=theta-r*(alpha/m)
        theta=np.array([theta[0,0],theta[0,1]])
    return(theta)

def computeCost(X,y,THETA):
    H=X.dot(THETA)
    l=np.array(H-y)**2
    j=np.sum(l)/(2*m)
    return(j)
   
def gradientDescentMulti():
    pass
def computeCostMulti():
    pass
def featureNormalize():
    pass
def normalEqn():
    pass
def predict(X):
    return np.array(X).dot(THETA)


data = open('ex1data1.txt', 'r').read().replace('\n',',').split(',')
m=len(data)//2
x,y=[float(data[i*2]) for i in range(m)],[float(data[i*2+1]) for i in range(m)]
#plotdata(x,y)
print(x)
#------------------------------ single only 
X=np.matrix( [ [1,i] for i in x ])
n=X[0].shape[1]
THETA=np.array( [0 for i in range(n)] ).transpose()
cost=computeCost(X,y,THETA)
iterations,alpha=1500,0.01
THETA = gradientDescent(X, y, THETA, alpha, iterations);
plotdata(x,y,True)

