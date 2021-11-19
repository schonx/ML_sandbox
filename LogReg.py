import numpy
import math
# Test
data_version = "1k" # "1k" or "100"

# import data
path = r"C:\Users\felix\sciebo2\Atom_working_dir\ML_sandbox\\" # put in global path here
x_logregdata = numpy.genfromtxt(path+"x_logregdata"+data_version+".csv", delimiter=",")
y_logregdata = numpy.genfromtxt(path+"y_logregdata"+data_version+".csv", delimiter=",")

class MyLogReg:

    def __init__(self, n):
        self._numinputs = n
        self._weights = numpy.repeat(1.0,n)
        self._bias = 0 # or = numpy.random.rand(1)
        self._lam = 0 # lambda parameter for regularization

    def pred(self, x):

        return sigmoid(numpy.matmul(x, self._weights) + self._bias)

# -----------------------------------------------------------------------------
# functions

def logloss(net,x,y): # p=prediction, t=target
    p = sigmoid(numpy.matmul(x, net._weights) + net._bias) # (1000,3)x(3,1) becomes (1000,1)
    logloss_term = (numpy.matmul(-y.T, numpy.log(p)) - numpy.matmul((1-y.T),numpy.log(1-p)))/len(p) # transpose, so (1,1000)x(1000,1) becomes (1)
    regularization_term = numpy.mean(net._weights**2 * net._lam/2)
    J = logloss_term + regularization_term
    return J

def sigmoid(z):
    # z = b + w1x1 + w2x2 + ... + wnxn
    return 1/(1+numpy.exp(-z))

# def gradient_Descent(LogRegObject, lr, x, y):
#     m = x.shape[0]
#     p = sigmoid(numpy.matmul(x, LogRegObject._weights))
#     grad = numpy.matmul(x.T, (p-y))/m
#     LogRegObject._weights -= lr * grad
#     LogRegObject._bias -= lr*numpy.mean(p-y)
#     return LogRegObject._weights

def train(net,x,t,epochs,lr):
    for epoch in range(epochs):
        # x comes in the shape of {n_samples, n_features}
        p = net.pred(x) # make predictions
        e = logloss(net,x,t)
        grad_w = (numpy.matmul(x.T,(p-t)) + net._lam*net._weights)/len(p) # dim: (3,1)
        grad_b = delta_bi = numpy.mean(p-t)
        if (epoch+1)%100 == 0:
            print(f"Epoch {epoch+1} | Error: {e} with weights {NN._weights} and bias {NN._bias}")
            print(f"grad_w: {grad_w} --- grad_b: {grad_b}")
        # update weights and bias
        NN._weights += lr*(-grad_w)
        NN._bias += lr*(-grad_b)

NN = MyLogReg(3)

# # -----------------------------------------------------------------------------
# # training process
print(80*"--")
print("Training process started!\n")
train(NN, x_logregdata, y_logregdata, 200, 0.3)

# for i in range(20):
#     print((NN.pred(x_logregdata[i,:]), y_logregdata[i]))


# Confusion matrix
p = (NN.pred(x_logregdata)>0.5).astype(int)
TP = 0
TN = 0
FP = 0
FN = 0
for i in range(len(p)):
    if p[i] == 1 and y_logregdata[i] == 1:
        TP += 1
    elif p[i] == 1 and y_logregdata[i] == 0:
        FP += 1
    elif p[i] == 0 and y_logregdata[i] == 1:
        FN += 1
    elif p[i] == 0 and y_logregdata[i] == 0:
        TN += 1

print(f"TP: {TP} FN: {FN}\n FP: {FP} TN: {TN}")
