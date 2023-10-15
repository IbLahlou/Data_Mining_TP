
import matplotlib.pyplot as plt
import numpy as np

def plot_data_model(X,y,bias,w1):
    plt.scatter(X,y,color="black")
    plt.plot(X,2+3*X,color="r") # model to find +/- (bias=2, w1=3)
    plt.plot(X,bias+w1*X,color="b") #trained model (initial model: bias=1, w1=0)
    plt.show()

def plot_data(X,y):
    plt.scatter(X,y,color="black")
    plt.plot(X,2+3*X,color="r") # model to find +/- (bias=2, w1=3)
    plt.show()

def mse(y,yhat):
    return 1/(len(y))*np.sum((y-yhat)**2)

#cette fonction est utilisé dans tout le tutorial
def tracer_graphes(w1List,mseList,iterations):
    plt.figure(figsize=(10,3))
    
    plt.subplot(121) # 1, 2,1
    plt.plot(w1List,mseList)
    plt.xlabel("w1")
    plt.ylabel("mse")
    plt.title("mse en fonction de w1")
   


    plt.subplot(122)
    plt.plot(iterations,mseList)

    plt.xlabel("iteration")
    plt.ylabel("mse")
    plt.title("mse en fonction des iterations")
