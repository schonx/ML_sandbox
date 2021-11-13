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
    return numpy.mean((p-t)**2)

def diffErr(net,x,p,t):
    # e = (y-t)**2
    # y = w*x+b
    # de/dw = de/dy*dy/dw = 2*(y-t) * x
    delta_wi = 2*(p-t)*x
    avg_dw = numpy.mean(delta_wi, axis=1) # sum over rows (=num of inputs)
    delta_bi = 2*(p-t)
    avg_db = numpy.mean(delta_bi) # sum all biases
    return numpy.array([avg_dw, avg_db], dtype=object)

def train(NN,x,t,epochs):
    n = NN._numinputs
    lr = 0.01
    for epoch in range(epochs):
        # give multiple inputs as x = numpy.array([x1,x2,...],dtype=object)
        p = NN.pred(x) # make predictions
        e = mse(p,t)
        grad_w, grad_b = diffErr(NN,x,p,t)
        print(f"Epoch {epoch} | Error: {e} with weights {NN._weights} and bias {NN._bias}")
        # update weights and bias
        NN._weights += lr*(-grad_w)
        NN._bias += lr*(-grad_b)



# def updateWB(net,delta_w,delta_b,h):
#     net._weight += delta_w * h
#     net._bias += delta_b * h
#     return net
#
# # -----------------------------------------------------------------------------
# # intialization
#
NN = FelixNet(2)
print(f"Initial weights: {NN.wb[0]} | initial bias: {NN.wb[1]}")
x1 = numpy.arange(0, 10, 0.1) # input values
x2 = numpy.arange(1, 11, 0.1) # input values
x = numpy.array([x1,x2])
t = 3*x1 + x2 + 2 # target values
p = NN.pred(x)
print(f"Predictions {p} for targets {t} were made.")
print(f"The error MSE is: {mse(p,t)}")

print(x.shape)
print(NN._weights.shape, NN._bias.shape)
p = NN.pred(x)

# print(type(numpy.array(x1[2],x2[2])))
print(20*"-")
dwb = diffErr(NN,x,p,t)
print(numpy.shape(dwb[0]),numpy.shape(dwb[1]))
print(dwb)

print(80*"-")
print("Training process started!\n")
train(NN,x,t,1000)
# h = 0.02 # learning rate
# epochs = 400
#
# # -----------------------------------------------------------------------------
# # application
#
# for i in range(epochs):
#     e = numpy.zeros(len(x))
#     wb = []
#
#     for j in x: # x is the "training data"
#         p = NN.pred(j) # make a prediction
#         e[j] = leastsquares(p, t[j]) # log the error of that prediction with regard to the actual target
#         wb.append(diffErr(NN,p,t[j],j)) # compute the negative gradient of the error for that example and log it
#
#     w_mean = numpy.mean([w[0] for w in wb]) # extract all logged weight gradients and take the mean
#     b_mean = numpy.mean([b[1] for b in wb]) # extract all logged bias gradients and take the mean
#
#     NN = updateWB(NN,w_mean,b_mean,h) # change weights and biases of the current model by learn_rate * delta
#     print(f"Epoch: {i+1} | Error: {numpy.mean(e)}, Weight: {NN.wb[0]}, Bias: {NN.wb[1]}")
#
# print(f"Final weight: {NN.wb[0]} | final bias: {NN.wb[1]}")
