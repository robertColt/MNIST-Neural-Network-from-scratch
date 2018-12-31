import numpy as np


class Layer():

    def __init__(self, n_units, activation_fct, derivative_act):
        self.n_units = n_units
        self.w = None
        self.b = None
        self.z = None
        self.a = None
        self.delta = None
        self.f = activation_fct
        self.df = derivative_act


    def feed_forward(self, X):
        self.z = np.matmul(self.w, X) + self.b
        self.a = self.f(self.z)


    def update(self, learn_rate, a):
        self.w = self.w - learn_rate*self.delta * a.T
        self.b = self.b - learn_rate*self.delta
