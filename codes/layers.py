"""
change log:
- Version 1: change the out_grads of `backward` function of `ReLU` layer into inputs_grads instead of in_grads
"""

import numpy as np 
from utils.tools import *

class Layer(object):
    """
    
    """
    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False # Whether there are parameters in this layer that can be trained

    def forward(self, inputs):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, in_grads, inputs):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradients of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
        """Initialization

        # Arguments
            in_features: int, the number of inputs features
            out_features: int, the numbet of required outputs features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_features)

        # Returns
            outputs: numpy array with shape (batch, out_features)
        """
        outputs = None
        #############################################################
        outputs = self.weights.T @ inputs.T + self.bias.reshape(-1,1)
        outputs = outputs.T
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_features), gradients to outputs
            inputs: numpy array with shape (batch, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_features), gradients to inputs
        """
        out_grads = None
        #############################################################
        batch_size = inputs.shape[0]
        x_rs = np.reshape(inputs, (batch_size, -1))
        self.b_grad = in_grads.sum(axis=0)
        self.w_grad = x_rs.T.dot(in_grads)
        dx = in_grads.dot(self.weights.T)
        out_grads = dx.reshape(inputs.shape)
        #############################################################
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v
        
    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class Convolution(Layer):
    def __init__(self, conv_params, initializer=Guassian(), name='conv'):
        """Initialization

        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border of padded with zeros.
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
            initializer: Initializer class, to initialize weights
        """
        super(Convolution, self).__init__(name=name)
        self.trainable = True
        self.kernel_h = conv_params['kernel_h'] # height of kernel
        self.kernel_w = conv_params['kernel_w'] # width of kernel
        self.pad = conv_params['pad']
        self.stride = conv_params['stride']
        self.in_channel = conv_params['in_channel']
        self.out_channel = conv_params['out_channel']

        self.weights = initializer.initialize((self.out_channel, self.in_channel, self.kernel_h, self.kernel_w))
        self.bias = np.zeros((self.out_channel))

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        outputs = None
        #############################################################
        p = self.pad
        h_out = 1 + int((inputs[0, 0].shape[0] + 2*p - self.kernel_h) / self.stride)
        w_out = 1 + int((inputs[0, 0].shape[1] + 2*p - self.kernel_w) / self.stride)

        x_padded = np.pad(inputs, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        N, C = inputs.shape[:2]
        i0 = np.repeat(np.arange(self.kernel_h), self.kernel_w)
        i0 = np.tile(i0, C)
        i1 = self.stride * np.repeat(np.arange(h_out), w_out)
        j0 = np.tile(np.arange(self.kernel_h), self.kernel_w * C)
        j1 = self.stride * np.tile(np.arange(h_out), w_out)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), self.kernel_h * self.kernel_w).reshape(-1, 1)

        X_col = x_padded[:, k, i, j]
        X_col = X_col.transpose(1, 2, 0).reshape(self.kernel_h * self.kernel_w * C, -1)

        W_col = self.weights.reshape(self.out_channel, -1)

        out = W_col @ X_col + self.bias.reshape(-1,1)
        out = out.reshape(self.out_channel, h_out, w_out, N)
        outputs = out.transpose(3, 0, 1, 2)
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        self.b_grad = np.sum(in_grads, axis=(0, 2, 3))
        
        p = self.pad
        h_out = 1 + int((inputs[0, 0].shape[0] + 2*p - self.kernel_h) / self.stride)
        w_out = 1 + int((inputs[0, 0].shape[1] + 2*p - self.kernel_w) / self.stride)

        x_padded = np.pad(inputs, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        N, C, H, W = inputs.shape
        i0 = np.repeat(np.arange(self.kernel_h), self.kernel_w)
        i0 = np.tile(i0, C)
        i1 = self.stride * np.repeat(np.arange(h_out), w_out)
        j0 = np.tile(np.arange(self.kernel_h), self.kernel_w * C)
        j1 = self.stride * np.tile(np.arange(h_out), w_out)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), self.kernel_h * self.kernel_w).reshape(-1, 1)

        X_col = x_padded[:, k, i, j]
        X_col = X_col.transpose(1, 2, 0).reshape(self.kernel_h * self.kernel_w * C, -1)

        in_grads_reshaped = in_grads.transpose(1, 2, 3, 0).reshape(self.out_channel, -1)
        self.w_grad = in_grads_reshaped.dot(X_col.T).reshape(self.weights.shape)

        dx_cols = self.weights.reshape(self.out_channel, -1).T.dot(in_grads_reshaped)
        # revert to image
        H_padded, W_padded = H + 2 * p, W + 2 * p
        x_padded_new = np.zeros((N, C, H_padded, W_padded), dtype=dx_cols.dtype)
        cols_reshaped = dx_cols.reshape(C * self.kernel_h * self.kernel_w, -1, N).transpose(2, 0, 1)
        np.add.at(x_padded_new, (slice(None), k, i, j), cols_reshaped)
        if p == 0:
            out_grads = x_padded_new
        else:
            out_grads = x_padded_new[:, :, p:-p, p:-p]
        #############################################################
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params
        
        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k,v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None

class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = np.maximum(0, inputs)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        inputs_grads = (inputs >=0 ) * in_grads
        out_grads = inputs_grads
        return out_grads


# TODO: add padding
class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_type = pool_params['pool_type']
        self.pool_height = pool_params['pool_height']
        self.pool_width = pool_params['pool_width']
        self.stride = pool_params['stride']
        self.pad = pool_params['pad']

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        outputs = None
        #############################################################
        p = self.pad
        st = self.stride
        inputs = np.pad(inputs, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
        p_h = self.pool_height
        p_w = self.pool_width
        batch, channel, in_h, in_w = inputs.shape
        out_h = 1 + int((in_h + 2 * p - p_h) // st)
        out_w = 1 + int((in_w + 2 * p - p_w) // st)
        outputs = np.zeros((batch, channel, out_h, out_w))
        self.location = np.zeros((batch, channel, out_h * out_w))

        x, y = self.get_coordinates(out_h, out_w)
        inputs_chunk = inputs[:, :, x, y]  # input_chunk is the all the data chunk extracted from inputs
        if self.pool_type == 'avg':
            inputs_sum = inputs_chunk.sum(axis=2)/(p_h*p_w)
            outputs = inputs_sum.reshape(batch, channel, out_h, out_w)
        elif self.pool_type == 'max':
            inputs_max = inputs_chunk.max(axis=2)
            outputs = inputs_max.reshape(batch, channel, out_h, out_w)
            self.location = inputs_chunk.argmax(axis=2)
        #############################################################
        return outputs
        
    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        out_grads = None
        #############################################################
        st = self.stride
        p_h = self.pool_height
        p_w = self.pool_width
        batch, channel, out_h, out_w = in_grads.shape
        in_h, in_w = inputs.shape[2:]
        out_grads = np.zeros((batch, channel, in_h, in_w))
        for b in range(batch):
            for c in range(channel):
                for h in range(0, out_h):
                    for w in range(0, out_w):
                        if self.pool_type == 'max':
                            max_h = h * st + math.floor(self.location[b, c, h*out_w+w]/p_w)
                            max_w = w * st + self.location[b, c, h*out_w+w]%p_w
                            out_grads[b, c, max_h, max_w] = in_grads[b, c, h, w]
                        elif self.pool_type == 'avg':
                            out_grads[b, c, h*st:h*st+p_h, w*st:w*st+p_w] = in_grads[b, c, h, w]
        #############################################################
        return out_grads

    def get_coordinates(self, h_out, w_out):

        x0 = np.repeat(np.arange(self.pool_height), self.pool_width)
        x1 = self.stride * np.repeat(np.arange(h_out), w_out)
        y0 = np.tile(np.arange(self.pool_height), self.pool_width)
        y1 = self.stride * np.tile(np.arange(h_out), w_out)
        x = x0.reshape(-1, 1) + x1.reshape(1, -1) # generate x is the x-coordinates of all the datapoints to be extracted
        y = y0.reshape(-1, 1) + y1.reshape(1, -1) # generate y is the x-coordinates of all the datapoints to be extracted

        return x, y

class Dropout(Layer):
    def __init__(self, ratio, name='dropout', seed=None):
        """Initialization

        # Arguments
            ratio: float [0, 1], the probability of setting a neuron to zero
            seed: int, random seed to sample from inputs, so as to get mask. (default as None)
        """
        super(Dropout, self).__init__(name=name)
        self.ratio = ratio
        self.mask = None
        self.seed = seed

    def forward(self, inputs):
        """Forward pass (Hint: use self.training to decide the phrase/mode of the model)

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = None
        #############################################################
        np.random.seed(self.seed)
        if self.training:
            if self.mask is None:
                self.mask = np.random.binomial(1, 1-self.ratio, size=inputs.shape)  
            inputs = inputs * self.mask * (1 / (1-self.ratio))
            outputs = inputs
        else:
            outputs = inputs
        #############################################################
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        out_grads = None
        #############################################################
        if self.training:
            out_grads = in_grads * self.mask * (1 / (1-self.ratio))
        else:
            out_grads = in_grads
        #############################################################
        return out_grads

class Flatten(Layer):
    def __init__(self, name='flatten', seed=None):
        """Initialization
        """
        super(Flatten, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel*in_height*in_width)
        """
        batch = inputs.shape[0]
        outputs = inputs.copy().reshape(batch, -1)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel*in_height*in_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs 
        """
        out_grads = in_grads.copy().reshape(inputs.shape)
        return out_grads
        
