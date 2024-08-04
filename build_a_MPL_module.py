import random
from function_for_neuron import Value

class Module:
    #Sets the gradient of all parameters to zero. This is useful to reset gradients before performing a new round of backpropagation.
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    # Returns an empty list. This method is intended to be overridden by subclasses to return their parameters.
    def parameters(self):
        return []
    
class Neuron(Module):
    #nin: Number of input connections.
    def __init__(self, nin, nonlin=True):#nonlin: Indicates whether to apply a non-linear activation function (ReLU) or not.
        #self.w is weight will be distributed
        self.w = [Value(random.uniform(-10, 10)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
    #compute and regtur= one neuron given the weights bias, and input(x) (sum of wi * xi + b)
    def __call__(self, x):
        # Ensure x is a list of Value instances
        assert isinstance(x, list) and all(isinstance(xi, Value) for xi in x), \
            "Input x should be a list of Value instances"
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        print(act)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        #def a a layer containing neurons
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        
    #let Neuron compute each neuron in one layer and put each neuron in a list and return it
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        print(out)
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
               
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts #total size = [3, 2, 3, 1]
        
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]
                      
        #[Layer of [ReLUNeuron(3), ReLUNeuron(3)], Layer of [ReLUNeuron(2), ReLUNeuron(2), ReLUNeuron(2)], Layer of [LinearNeuron(3)]]
    #get the final output from the input x.
    def __call__(self, x):
        for layer in self.layers:
            #use x to store the output of the layer and use it to compute the next layer, that was just great!!!
            x = layer(x)
            
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

# Create an MLP with 3 inputs, a hidden layer with 4 neurons, and an output layer with 1 neuron
#The first 3 stands for number of inputs, 2stands for the first layer have 2 neurons, 3 stands for the second layers have 3 neurons, 
#and 1 is the output layer has 1 neuron
mlp = MLP(4, [2,3,1]) 

# Create an input vector
x = [Value(3.0), Value(2.0), Value(3.0)]

# Get the output of the MLP for the input vector
output = mlp(x)

# Print the output
print(output)

# Print the parameters of the MLP
print(mlp.parameters())