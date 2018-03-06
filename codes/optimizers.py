"""
changelog:
- version 2: remove bias_correction in the comments of Adam (no need to implement it);
                     correct the implementation of RMSprop (self.accumulators[k] = self.rho * self.accumulators[k] + (1 - self.rho) * xs_grads[k]**2)
"""
import numpy as np
import copy

class Optimizer():
    
    def __init__(self, lr):
        """Initialization
        
        # Arguments
            lr: float, learnig rate 
        """
        self.lr = lr

    def update(self, x, x_grad, iteration):
        """Update parameters with gradients"""
        raise NotImplementedError

    def sheduler(self, func, iteration):
        """learning rate sheduler, to change learning rate with respect to iteration
        
        # Arguments
            func: function, arguments are lr and iteration
            iteration: int, current iteration number in the whole training process (not in that epoch)
        
        # Returns
            lr: float, the new learning rate
        """
        lr = func(self.lr, iteration)
        return lr

class SGD(Optimizer):
    
    def __init__(self, lr=0.01, momentum=0, decay=0, sheduler_func = None):
        """Initialization
        
        # Arguments
            lr: float, learnig rate 
            momentum: float, the ratio of moments
            decay: float, the learning rate decay ratio
        """
        super(SGD, self).__init__(lr)
        self.momentum = momentum
        self.moments = None
        self.decay = decay
        self.sheduler_func = sheduler_func

    def update(self, xs, xs_grads, iteration):
        """Initialization
        
        # Arguments
            xs: dictionary, all weights of model
            xs_grads: dictionary, gradients to all weights of model, same keys with xs
            iteration: int, current iteration number in the whole training process (not in that epoch)

        # Returns
            new_xs: dictionary, new weights of model
        """
        new_xs = {}
        if self.decay > 0:
            self.lr *= (1/(1+self.decay*iteration))
        if self.sheduler_func:
            self.lr = self.sheduler(self.sheduler_func, iteration)

        # initialize self.moments
        if not self.moments:
            self.moments = {}
            for k, v in xs_grads.items():
                self.moments[k] = np.zeros(v.shape)

        prev_moments = copy.deepcopy(self.moments)

        for k in list(xs.keys()):
        #############################################################
            v = self.momentum * self.moments[k] - self.lr * xs_grads[k]
            new_xs[k] = xs[k] - self.lr * xs_grads[k] + self.momentum * v
        #############################################################
        return new_xs

class Adam(Optimizer):
    
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, sheduler_func=None):
        """Initialization
        
        # Arguments
            lr: float, learnig rate 
            beta_1: float
            beta_2: float
            epsilon: float, precision to avoid numerical error
            decay: float, the learning rate decay ratio
        """
        super(Adam, self).__init__(lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        if not self.epsilon:
            self.epsilon = 1e-8

        self.moments = None
        self.accumulators = None
        self.sheduler_func = sheduler_func

    def update(self, xs, xs_grads, iteration):
        """Initialization
        
        # Arguments
            xs: dictionary, all weights of model
            xs_grads: dictionary, gradients to all weights of model, same keys with xs
            iteration: int, current iteration number in the whole training process (not in that epoch)

        # Returns
            new_xs: dictionary, new weights of model
        """
        new_xs = {}
        if self.decay > 0:
            self.lr *= (1/(1+self.decay*iteration))
        if self.sheduler_func:
            self.lr = self.sheduler(self.sheduler_func, iteration)
        # initialization of moments and accumulators
        if not self.accumulators or self.moments:
            self.moments = {}
            self.accumulators = {}
            for k,v in xs.items():
                self.moments[k] = np.zeros(v.shape)
                self.accumulators[k] = np.zeros(v.shape)

        for k in list(xs.keys()):
        #############################################################
            self.moments[k] = self.beta_1*self.moments[k] + (1-self.beta_1)*xs_grads[k]
            self.accumulators[k] = self.beta_2 * self.accumulators[k] + (1 - self.beta_2) * (xs_grads[k] ** 2)
            new_xs[k] = xs[k] - self.lr * self.moments[k] / (np.sqrt(self.accumulators[k]) + self.epsilon)
        #############################################################
        return new_xs

class Adagrad(Optimizer):
    def __init__(self, lr=0.01, epsilon=None, decay=0, sheduler_func=None):
        """Initialization
        
        # Arguments
            lr: float, learnig rate 
            epsilon: float, precision to avoid numerical error
            decay: float, the learning rate decay ratio
        """
        super(Adagrad, self).__init__(lr)
        self.epsilon = epsilon
        self.decay = decay
        if not self.epsilon:
            self.epsilon = 1e-8
        self.accumulators = None
        self.sheduler_func = sheduler_func

    def update(self, xs, xs_grads, iteration):
        """Initialization
        
        # Arguments
            xs: dictionary, all weights of model
            xs_grads: dictionary, gradients to all weights of model, same keys with xs
            iteration: int, current iteration number in the whole training process (not in that epoch)

        # Returns
            new_xs: dictionary, new weights of model
        """
        new_xs = {}
        if self.decay > 0:
            self.lr *= (1/(1+self.decay*iteration))
        if self.sheduler_func:
            self.lr = self.sheduler(self.sheduler_func, iteration)
        if not self.accumulators:
            self.accumulators = {}
            for k,v in xs.items():
                self.accumulators[k] = np.zeros(v.shape)
        for k in list(xs.keys()):
            self.accumulators[k] += xs_grads[k]**2
            new_xs[k] = xs[k] - self.lr * xs_grads[k] / (np.sqrt(self.accumulators[k]) + self.epsilon)
        return new_xs

class RMSprop(Optimizer):
    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0, sheduler_func=None):
        """Initialization
        
        # Arguments
            lr: float, learnig rate 
            rho: float
            epsilon: float, precision to avoid numerical error
            decay: float, the learning rate decay ratio
        """
        super(RMSprop, self).__init__(lr)
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay
        if not self.epsilon:
            self.epsilon = 1e-8
        self.accumulators = None
        self.sheduler_func = sheduler_func

    def update(self, xs, xs_grads, iteration):
        """Initialization
        
        # Arguments
            xs: dictionary, all weights of model
            xs_grads: dictionary, gradients to all weights of model, same keys with xs
            iteration: int, current iteration number in the whole training process (not in that epoch)

        # Returns
            new_xs: dictionary, new weights of model
        """
        new_xs = {}
        if self.decay > 0:
            self.lr *= (1/(1+self.decay*iteration))
        if self.sheduler_func:
            self.lr = self.sheduler(self.sheduler_func, iteration)
        if not self.accumulators:
            self.accumulators = {}
            for k,v in xs.ietms():
                self.accumulators[k] = np.zeros(v.shape)
        for k in list(xs.keys()):
            self.accumulators[k] = self.rho * self.accumulators[k] + (1 - self.rho) * xs_grads[k]**2
            new_xs[k] = xs[k] - self.lr * xs_grads[k] / (np.sqrt(self.accumulators[k]) + self.epsilon)
        return new_xs