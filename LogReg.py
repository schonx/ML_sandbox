import numpy
import math

# import data
path = r"C:\Users\felix\sciebo2\Atom_working_dir\ML_sandbox\\" # put in global path here
x_logregdata = numpy.genfromtxt(path+"x_logregdata.csv", delimiter=",")
y_logregdata = numpy.genfromtxt(path+"y_logregdata.csv", delimiter=",")

class MyLogReg:

    def __init__(self, n):
        self._numinputs = n
        self._weights = numpy.random.rand(n)
        self._bias = 0 # or = numpy.random.rand(1)
        self._lam = 0.001 # lambda parameter for regularization

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

def gradient_Descent(LogRegObject, lr, x, y):
    m = x.shape[0]
    p = sigmoid(numpy.matmul(x, LogRegObject._weights))
    grad = numpy.matmul(x.T, (p-y))/m
    LogRegObject._weights -= lr * grad
    LogRegObject._bias -= lr*numpy.mean(p-y)
    return LogRegObject._weights

def train(net,x,t,epochs,lr):
    for epoch in range(epochs):
        # x comes in the shape of {n_samples, n_features}
        p = net.pred(x) # make predictions
        e = logloss(net,x,t)
        grad_w = (numpy.dot(x.T,(p-t)) + net._lam*net._weights)/len(p)
        grad_b = delta_bi = numpy.mean(p-t)
        if (epoch+1)%100 == 0:
            print(f"Epoch {epoch+1} | Error: {e} with weights {NN._weights} and bias {NN._bias}")
            print(f"grad_w: {grad_w} --- grad_b: {grad_b}")
        # update weights and bias
        NN._weights += lr*(-grad_w)
        NN._bias += lr*(-grad_b)

# -----------------------------------------------------------------------------
# DEPRECATED
# data from https://datatofish.com/logistic-regression-python/
# helpful: https://towardsdatascience.com/implement-logistic-regression-with-l2-regularization-from-scratch-in-python-20bd4ee88a59

# candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
#               'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
#               'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
#               'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
#               }
# x_gmat = numpy.asarray(candidates['gmat'])
# x_gpa = numpy.asarray(candidates['gpa'])
# x_we = numpy.asarray(candidates['work_experience'])
# y_adm = numpy.asarray(candidates['admitted'])
# x_combined = numpy.array([x_gmat, x_gpa, x_we])
# -----------------------------------------------------------------------------


NN = MyLogReg(3)

n_iterations = 500
learning_rate = 0.5

# for i in range(n_iterations):
#     Theta = gradient_Descent(NN, learning_rate, x_logregdata, y_logregdata)
#     if i % 50 == 0:
#         print(logloss(NN, x_logregdata, y_logregdata))

# NN._weights = Theta
# print(NN._weights, NN._bias)
# print(NN.pred(x_logregdata[:5,:]), y_logregdata[:5])

# # -----------------------------------------------------------------------------
# # training process
print(80*"--")
print("Training process started!\n")
train(NN, x_logregdata, y_logregdata, 1000, 0.1)
# print(numpy.shape(x_logregdata))
# print(numpy.matmul(x_logregdata, NN._weights))
# # test against test_data
# print(80*"--")
# x = x_logregdata[:5,:]
# p = NN.pred(x)
# t = y_logregdata[:5]
# print(p,t)
# delta_wi = (numpy.matmul(x.T,(p-t)))/len(p)
# print(delta_wi)
# print(numpy.shape(p-t))
# print(numpy.shape(x.T))
# print(numpy.shape(x_logregdata))
# print(numpy.shape(NN._weights))
