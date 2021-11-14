import numpy
import math

# import data
path = r"C:\Users\felix\sciebo2\Atom_working_dir\ML_sandbox\\" # put in global path here
x_logregdata = numpy.genfromtxt(path+"x_logregdata.csv", delimiter=",")
y_logregdata = numpy.genfromtxt(path+"y_logregdata.csv", delimiter=",")

class FelixLog:

    def __init__(self, n):
        self._numinputs = n
        #self._weights = numpy.random.rand(n)
        self._weights = numpy.zeros(n)
        self._bias = numpy.random.rand(1)
        self._lam = 0.001 # lambda parameter for regularization

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

        return sigmoid(numpy.matmul(x, self._weights) + self._bias)

# -----------------------------------------------------------------------------
# functions

def logloss(net,p,t): # p=prediction, t=target
    logloss_term = numpy.mean(-t*numpy.log(p)-(1-t)*numpy.log(1-p))
    regularization_term = numpy.mean(net._weights**2 * net._lam/2)
    J = logloss_term + regularization_term
    return J

def sigmoid(z):
    # z = b + w1x1 + w2x2 + ... + wnxn
    return 1/(1+numpy.exp(-z))

def calcgrad(net,x,p,t): # same update rule as in linear regression
    delta_wi = numpy.matmul((p-t),x) + net._lam*net._weights
    delta_bi = numpy.mean(p-t)

    return numpy.array([delta_wi, delta_bi], dtype=object)

def train(NN,x,t,epochs):
    lr = 0.0001
    for epoch in range(epochs):
        # give multiple inputs as x = numpy.array([x1,x2,...],dtype=object)
        p = NN.pred(x) # make predictions
        e = logloss(NN,p,t)
        grad_w, grad_b = calcgrad(NN,x,p,t)
        if (epoch+1)%10 == 0:
            print(f"Epoch {epoch+1} | Error: {e} with weights {NN._weights} and bias {NN._bias}")
            print(f"grad_w: {grad_w} --- grad_b: {grad_b}")
        # update weights and bias
        NN._weights += lr*(-grad_w)
        NN._bias += lr*(-grad_b)

# -----------------------------------------------------------------------------
# DEPRECATED
# data stolen from https://datatofish.com/logistic-regression-python/
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


NN = FelixLog(3)

# p = NN.pred(x_logregdata[:,:10])
# print(p)
# # -----------------------------------------------------------------------------
# # training process
print(80*"--")
print("Training process started!\n")
train(NN, x_logregdata, y_logregdata, 1000)
#
# # test against test_data
# print(80*"--")
# print("Evaluation on test data:\n")
