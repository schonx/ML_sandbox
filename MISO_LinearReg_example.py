import numpy

class FelixNet:

    def __init__(self, n):
        self._numinputs = n
        self._weights = numpy.random.rand(n)
        self._bias = numpy.random.rand(1)

    @property
    def wb(self):
        return (self._weights, self._bias)

    @wb.setter
    def wb(self, weights, bias):
        try:
            self._weights[:] = weights
            self._bias = bias
        except Exception as e:
            raise

    def pred(self, x):

        return numpy.matmul(self._weights, x) + self._bias

# -----------------------------------------------------------------------------
# functions

def mse(p,t): # p=prediction, t=target
    return numpy.mean(1/2*(p-t)**2)

def diffErr(net,x,p,t):
    # e = (y-t)**2
    # y = w*x+b
    # de/dw = de/dy*dy/dw = 2*(y-t) * x
    delta_wi = (p-t)*x
    avg_dw = numpy.mean(delta_wi, axis=1) # mean over rows (=num of inputs)
    delta_bi = (p-t)
    avg_db = numpy.mean(delta_bi) # mean all biases
    return numpy.array([avg_dw, avg_db], dtype=object)
    # return numpy.array([delta_wi, delta_bi], dtype=object)

def train(NN,x,t,epochs):
    lr = 0.001
    for epoch in range(epochs):
        # give multiple inputs as x = numpy.array([x1,x2,...],dtype=object)
        p = NN.pred(x) # make predictions
        e = mse(p,t)
        grad_w, grad_b = diffErr(NN,x,p,t)
        print(f"Epoch {epoch} | Error: {e} with weights {NN._weights} and bias {NN._bias}")
        # update weights and bias
        NN._weights += lr*(-grad_w)
        NN._bias += lr*(-grad_b)
        # for i in range(x.shape[1]):
        #
        #     NN._weights += lr*(-grad_w[:,i])
        #     NN._bias += lr*(-grad_b[i])

# # -----------------------------------------------------------------------------
# # intialization
#
NN = FelixNet(2)
print(f"Initial weights: {NN.wb[0]} | initial bias: {NN.wb[1]}")
x1 = numpy.arange(0, 10, 0.5) # input values
x2 = numpy.arange(1, 11, 0.5) # input values
x = numpy.array([x1,x2])
t = 5*x1 + 3*x2 + 2 # target values
p = NN.pred(x)
print(f"Predictions {p} for targets {t} were made.")
print(f"The error MSE is: {mse(p,t)}")

print(f"input shape is {x.shape}")
print(f"weights shape is {NN._weights.shape} and bias shape is {NN._bias.shape}")
p = NN.pred(x)

# print(type(numpy.array(x1[2],x2[2])))
print(80*"-")
dwb = diffErr(NN,x,p,t)
print(numpy.shape(dwb[0]),numpy.shape(dwb[1]))
print(dwb[0],dwb[1])

print(80*"-")
print("Training process started!\n")
print(x.shape[1])
train(NN,x,t,1000)
# h = 0.02 # learning rate
# epochs = 400
