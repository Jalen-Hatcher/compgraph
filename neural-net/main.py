#import math 
#import matplotlib.pyplot as plt
#import numpy as np 
import graphify as gfy
import value as v
import nn

if __name__ == "__main__":
    '''# inputs to the neuron
    x1 = v.Value(2.0, label='x1')
    x2 = v.Value(0.0, label='x2')

    # weights of the neuron
    w1 = v.Value(-3.0, label='w1')
    w2 = v.Value(1.0, label='w2')

    # bias of the neuron
    b = v.Value(6.8813735870195432, label='b')

    x1w1 = x1*w1; x1w1.label='x1*w1'
    x2w2 = x2*w2; x2w2.label='x2*w2'

    x1w1_x2w2 = x1w1 + x2w2; x1w1_x2w2.label='x1*w1 + x2w2'
    n = x1w1_x2w2 + b; n.label = 'n'
    o = n.tanh(); o.label='o'

    o.backward()
    gfy.render(o)'''

    x = [2.0, 3.0, -1.0]
    n = nn.MLP(3, [4,4,1])
    gfy.render(n(x))

    
    
    