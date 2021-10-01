import numpy as np
import scipy.io
import sklearn
import matplotlib.pyplot as plt


def plot(X,y,model=False,clear=False):
    plt.close()#clear any other plot
    type_0=np.matrix( [(X[i,0],X[i,1]) for i in range(y.shape[0]) if not y[i]] )
    type_1=np.matrix( [(X[i,0],X[i,1]) for i in range(y.shape[0]) if y[i]] ) #could replace with a set or np.unique? 
    plt.plot(type_0[:,0],type_0[:,1], 'yo')
    plt.plot(type_1[:,0],type_1[:,1], 'kx')
    if type(model)==sklearn.svm._classes.SVC: # im sure you can improve this
        l1 = np.array([min(X[:,0]),max(X[:,0])])
        l2=-(model.coef_[0,0]*l1 +model.intercept_  )/model.coef_[0,1] #l2=(model.coef_[0,1] + model.coef_[0,0]*l1)
        plt.plot(l1, l2, '-b')
    
    
def gaussianKernel(x1, x2, sigma):
    temp=np.matrix(x1-x2)
    return np.exp( (temp*temp.transpose() )/(-2*(sigma**2) ))

def  gaussianKernelGramMatrix(X1,X2,sigma): # this only works with two features :( maybe add recursion and enmurate is 
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0])) # ur imlementation sucked buddy 
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = gaussianKernel(x1, x2, sigma)
    return gram_matrix

c = 100
Data=scipy.io.loadmat('ex6data1.mat')
X,y=Data['X'],np.matrix(Data['y']).A1
model = sklearn.svm.SVC(C=c, kernel="linear", tol=1e-3).fit(X,y)#,max_iter=100
plot(X,y,model)
plt.show()
sim=gaussianKernel(np.matrix([1,2,1]), np.matrix([0, 4, -1]) , 2) #0.324652
 
Data=scipy.io.loadmat('ex6data2.mat')
X,y=Data['X'],np.matrix(Data['y']).A1


c = 1
sigma=0.1
'''model = sklearn.svm.SVC(C = c, kernel="precomputed", tol=1e-3).fit( gaussianKernelGramMatrix(X,X,sigma) ,y)
plot(X,y)

x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T 
X1, X2 = np.meshgrid(x1plot, x2plot)
vals = np.zeros(X1.shape)
for i in range(X1.shape[1]):
       this_X = np.column_stack((X1[:, i], X2[:, i]))
       vals[:, i] = model.predict(gaussianKernelGramMatrix(this_X, X,sigma))

plt.contour(X1, X2, vals, colors="blue", levels=[0,0])
plt.show()



Data=scipy.io.loadmat('ex6data3.mat')
X,y=Data['X'],np.matrix(Data['y']).A1
model = sklearn.svm.SVC(C = c, kernel="precomputed", tol=1e-3).fit( gaussianKernelGramMatrix(X,X,sigma) ,y)
plot(X,y)

x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T 
X1, X2 = np.meshgrid(x1plot, x2plot)
vals = np.zeros(X1.shape)
for i in range(X1.shape[1]):
       this_X = np.column_stack((X1[:, i], X2[:, i]))
       vals[:, i] = model.predict(gaussianKernelGramMatrix(this_X, X,sigma))

plt.contour(X1, X2, vals, colors="blue", levels=[0,0])
plt.show()'''

