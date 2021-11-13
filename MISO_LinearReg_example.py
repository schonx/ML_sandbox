import numpy

class FelixNet:

    def __init__(self, n):
        self._numinputs = n
        self._weights = numpy.random.rand(n)
        self._bias = numpy.random.rand(1)

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

        return numpy.matmul(self._weights, x) + self._bias

    def eval(self,x,t):

        p = numpy.matmul(self._weights, x) + self._bias
        e = mse(p,t)
        print(f"Error is {e}.")
        return
# -----------------------------------------------------------------------------
# functions

def mse(p,t): # p=prediction, t=target
    return numpy.mean(1/2*(p-t)**2)

def diffErr(net,x,p,t):
    # e = 1/2* (y-t)**2
    # y = w*x+b
    # de/dw = de/dy*dy/dw = (y-t) * x
    delta_wi = (p-t)*x
    avg_dw = numpy.mean(delta_wi, axis=1) # mean over rows (=num of inputs)
    delta_bi = (p-t)
    avg_db = numpy.mean(delta_bi) # mean all biases
    return numpy.array([avg_dw, avg_db], dtype=object)

def train(NN,x,t,epochs):
    lr = 0.01
    for epoch in range(epochs):
        # give multiple inputs as x = numpy.array([x1,x2,...],dtype=object)
        p = NN.pred(x) # make predictions
        e = mse(p,t)
        grad_w, grad_b = diffErr(NN,x,p,t)
        print(f"Epoch {epoch+1} | Error: {e} with weights {NN._weights} and bias {NN._bias}")
        print(f"grad_w: {grad_w} --- grad_b: {grad_b}")
        # update weights and bias
        NN._weights += lr*(-grad_w)
        NN._bias += lr*(-grad_b)


# # -----------------------------------------------------------------------------
# intialization
NN = FelixNet(2)
print(f"Initial weights: {NN.wb[0]} | initial bias: {NN.wb[1]}")

# make a grid
xnew1 = numpy.linspace(0,5,21)
xnew2 = numpy.linspace(0,5,21)
xv,yv = numpy.meshgrid(xnew1,xnew2)
# evaluate a linear function on that grid
tnew = 7*xv + 2*yv + 4

# transform data into 1-d arrays
x1 = numpy.reshape(xv, len(xv)**2)
x2 = numpy.reshape(yv, len(yv)**2)
input = numpy.array([x1,x2])
t = numpy.reshape(tnew, len(tnew)**2)

# shuffle data
rng = numpy.random.default_rng()
collector = numpy.array([x1,x2,t])
rng.shuffle(collector, axis=1)

# separate training (2/3) and test (1/3) data
n = numpy.shape(collector)[1]
train_data = collector[:,:int(2/3*n)]
test_data = collector[:,int(2/3*n):]
train_x1, train_x2, train_t = train_data
test_x1, test_x2, test_t = test_data

# training process
print(80*"--")
print("Training process started!\n")
train(NN,numpy.array([train_x1,train_x2]),train_t,2000)

# test against test_data
print(80*"--")
print("Evaluation on test data:\n")
NN.eval(numpy.array([test_x1,test_x2]),test_t)
