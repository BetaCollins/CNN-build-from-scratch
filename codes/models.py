import numpy as np 
import copy, pickle, sys
from utils.tools import clip_gradients

class Model():
    
    def __init__(self):
        self.layers = []
        self.inputs = None
        self.optimizer = None 
        self.regularization = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss, regularization=None):
        self.optimizer = optimizer
        self.layers.append(loss)
        self.regularization = regularization

    def forward(self, inputs, targets):
        self.inputs = []
        layer_inputs = inputs
        for l, layer in enumerate(self.layers):
            # print(layer.name, layer_inputs)
            self.inputs.append(layer_inputs)
            if l==len(self.layers)-1:
                layer_inputs, probs = layer.forward(layer_inputs, targets)
            else:
                layer_inputs = layer.forward(layer_inputs)
        outputs = layer_inputs
        return outputs, probs

    def backward(self, targets):
        for l, layer in enumerate(self.layers[::-1]):
            if l==0:
                grads = layer.backward(self.inputs[-1-l], targets)
            else:
                grads = layer.backward(grads, self.inputs[-1-l])

    def get_params(self):
        params = {}
        grads = {}
        for l, layer in enumerate(self.layers):
            if layer.trainable:
                layer_params, layer_grads = layer.get_params('layer-%dth'%l)
                params.update(layer_params)
                grads.update(layer_grads)

        if self.regularization:
            reg_grads = self.regularization.backward(params)
            for k, v in grads.items():
                grads[k] += reg_grads[k]
        return params, grads

    def update(self, optimizer, iteration):
        params, grads = self.get_params()

        # clip gradients
        # for k, v in grads.items():
        #     grads[k] = clip_gradients(v)
        #     print(k, np.mean(np.abs(v)))

        new_params = optimizer.update(params, grads, iteration)

        for l, layer in enumerate(self.layers):
            if layer.trainable:
                w_key = 'layer-%dth:'%l + layer.name + '/weights'
                b_key = 'layer-%dth:'%l + layer.name + '/bias'
                layer_params = {
                    w_key: new_params[w_key],
                    b_key: new_params[b_key]
                }
                layer.update(layer_params)

    def train(self, dataset, train_batch=32, val_batch=1000, test_batch=1000, epochs=5, val_intervals=100, test_intervals=500, print_intervals=100):
        train_loader = dataset.train_loader(train_batch)
        num_train = dataset.num_train

        train_results = []
        test_results = []
        val_results = []

        for epoch in range(epochs):
            print('Epoch %d: '%epoch, end='\n')
            for iteration in range(num_train//train_batch):
                
                total_iteration = epoch*(num_train//train_batch)+iteration
                # output test loss and accuracy
                if iteration % test_intervals == 0:
                    test_loss, test_acc = self.test(dataset, test_batch)
                    test_results.append([total_iteration, test_loss, test_acc])


                if iteration % val_intervals == 0:
                    val_loss, val_acc = self.val(dataset, val_batch)
                    val_results.append([total_iteration, val_loss, val_acc])

                x, y = next(train_loader)
                loss, probs = self.forward(x, y)
                acc = np.sum(np.argmax(probs, axis=-1)==y) / train_batch
                train_results.append([total_iteration, loss, acc])

                if self.regularization:
                    params, _ = self.get_params()
                    reg_loss = self.regularization.forward(params)

                if iteration % print_intervals == 0:
                    print('Iteration %d:\t'%iteration, end='')
                    print('accuracy=%.5f, loss=%.5f'%(acc, loss), end='')
                    if self.regularization:
                        print(', regularization loss=', reg_loss)
                    else:
                        print('\n')

                    # for layer in self.layers:
                    #     if layer.trainable:
                    #         print(layer.name, np.mean(np.abs(layer.weights)))
                
                self.backward(y)
                self.update(self.optimizer, total_iteration)
        return np.array(train_results), np.array(val_results), np.array(test_results)


    def test(self, dataset, test_batch):
        # set the mode into testing mode
        for layer in self.layers:
            layer.set_mode(training=False)
        test_loader = dataset.test_loader(test_batch)
        num_test = dataset.num_test
        num_accurate = 0
        sum_loss = 0
        try:
            while True:
                x, y = next(test_loader)
                loss, probs = self.forward(x, y)
                num_accurate += np.sum(np.argmax(probs, axis=-1)==y)
                sum_loss += loss
        except StopIteration:
            avg_loss = sum_loss*test_batch/num_test
            accuracy = num_accurate/num_test
            print('Test accuracy=%.5f, loss=%.5f'%(accuracy, avg_loss))

        # reset the mode into training for continous training
        for layer in self.layers:
            layer.set_mode(training=True)

        return avg_loss, accuracy
        

    
    def val(self, dataset, val_batch):
        # set the mode into testing mode
        for layer in self.layers:
            layer.set_mode(training=False)
        val_loader = dataset.val_loader(val_batch)
        num_val = dataset.num_val
        num_accurate = 0
        sum_loss = 0
        try:
            while True:
                x, y = next(val_loader)
                loss, probs = self.forward(x, y)
                num_accurate += np.sum(np.argmax(probs, axis=-1)==y)
                sum_loss += loss
        except StopIteration:
            avg_loss = sum_loss*val_batch/num_val
            accuracy = num_accurate/num_val
            print('Validation accuracy: %.5f, loss: %.5f'%(accuracy, avg_loss))

        # reset the mode into training for continous training
        for layer in self.layers:
            layer.set_mode(training=True)

        return avg_loss, accuracy
