import numpy

class FelixNet:

    def __init__(self):

        self._weight = numpy.random.randint(0, high=10)
        self._bias = numpy.random.randint(0, high=10)

    @property
    def wb(self):
        return (self._weight, self._bias)

    @wb.setter
    def wb(self, weight, bias):
        self._weight = weight
        self._bias = bias

    def pred(self, x):
        return self._weight * x + self._bias

# -----------------------------------------------------------------------------
# functions

def leastsquares(y,t):
    return (y-t)**2

def diffErr(net,y,t,xi):
    # e = (y-t)**2
    # y = w*x+b
    # de/dw = de/dy*dy/dw = 2*(y-t) * x
    delta_wi = -2*(y-t)*xi
    delta_bi = -2*(y-t)
    return [delta_wi, delta_bi]

def updateWB(net,delta_w,delta_b,h):
    net._weight += delta_w * h
    net._bias += delta_b * h
    return net

# -----------------------------------------------------------------------------
# intialization

NN = FelixNet()
print(f"Initial weight: {NN.wb[0]} | initial bias: {NN.wb[1]}")
x = numpy.arange(0, 10, 1) # input values
t = 3*x + 2 # target values
h = 0.02 # learning rate
epochs = 400

# -----------------------------------------------------------------------------
# application

for i in range(epochs):
    e = numpy.zeros(len(x))
    wb = []

    for j in x: # x is the "training data"
        p = NN.pred(j) # make a prediction
        e[j] = leastsquares(p, t[j]) # log the error of that prediction with regard to the actual target
        wb.append(diffErr(NN,p,t[j],j)) # compute the negative gradient of the error for that example and log it

    w_mean = numpy.mean([w[0] for w in wb]) # extract all logged weight gradients and take the mean
    b_mean = numpy.mean([b[1] for b in wb]) # extract all logged bias gradients and take the mean

    NN = updateWB(NN,w_mean,b_mean,h) # change weights and biases of the current model by learn_rate * delta
    print(f"Epoch: {i+1} | Error: {numpy.mean(e)}, Weight: {NN.wb[0]}, Bias: {NN.wb[1]}")

print(f"Final weight: {NN.wb[0]} | final bias: {NN.wb[1]}")
