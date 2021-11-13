import numpy
import math

class FelixLog:

    def __init__(self, n):
        self._numinputs = n
        #self._weights = numpy.random.rand(n)
        self._weights = numpy.zeros(n)
        self._bias = numpy.random.rand(1)
        self._lam = 0.01 # lambda parameter for regularization

    @property
    def wb(self): # getter fun
        return (self._weights, self._bias)

    @wb.setter
    def wb(self, weights, bias): # setter fun
        try:
            self._weights[:] = weights
            self._bias = bias
        except Exception as e:
            raise

    def pred(self, x):

        return sigmoid(numpy.matmul(self._weights, x) + self._bias)

    def eval(self,x,t):

        p = numpy.matmul(self._weights, x) + self._bias
        e = mse(p,t)
        print(f"Error is {e}.")
        return
# -----------------------------------------------------------------------------
# functions

def logloss(net,p,t): # p=prediction, t=target
    L2 = numpy.sum(net._weights**2, axis=0)
    return numpy.sum(-t*numpy.log(p)-(1-t)*numpy.log(1-p)) + net._lam/2/len(t)*L2

def sigmoid(z):
    # z = b + w1x1 + w2x2 + ... + wnxn
    return 1/(1+numpy.exp(-z))

def calcgrad(net,x,p,t): # same update rule as in linear regression
    delta_wi = (p-t)*x # + net._lam*net._weights
    avg_dw = numpy.mean(delta_wi, axis=1) # mean over rows (=num of inputs)
    delta_bi = (p-t)
    avg_db = numpy.mean(delta_bi) # mean all biases
    return numpy.array([avg_dw, avg_db], dtype=object)

def train(NN,x,t,epochs):
    lr = 0.000001
    for epoch in range(epochs):
        # give multiple inputs as x = numpy.array([x1,x2,...],dtype=object)
        p = NN.pred(x) # make predictions
        e = logloss(NN,p,t)
        grad_w, grad_b = calcgrad(NN,x,p,t)
        # print(f"Epoch {epoch+1} | Error: {e} with weights {NN._weights} and bias {NN._bias}")
        # print(f"grad_w: {grad_w} --- grad_b: {grad_b}")
        # update weights and bias
        NN._weights += lr*(-grad_w)
        NN._bias += lr*(-grad_b)

# data stolen from https://datatofish.com/logistic-regression-python/

candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }
x_gmat = numpy.asarray(candidates['gmat'])
x_gpa = numpy.asarray(candidates['gpa'])
x_we = numpy.asarray(candidates['work_experience'])
y_adm = numpy.asarray(candidates['admitted'])


NN = FelixLog(3)
x_combined = numpy.array([x_gmat, x_gpa, x_we])
train(NN, x_combined, y_adm, 1000)

p = NN.pred(numpy.array([x_gmat, x_gpa, x_we]))
print(f"prediction {p} with {numpy.shape(p)}")
print(f"y {y_adm} with {numpy.shape(y_adm)}")
print(f"x {x_combined} with {numpy.shape(x_combined)}")
print(100*"-")
print(numpy.matmul(x_combined,(p-y_adm)))
# # -----------------------------------------------------------------------------
# # training process
# print(80*"--")
# print("Training process started!\n")
#
#
# # test against test_data
# print(80*"--")
# print("Evaluation on test data:\n")
