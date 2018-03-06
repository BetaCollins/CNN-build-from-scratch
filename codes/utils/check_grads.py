import numpy as np 


# The first 3 functions in this file are from the Stanford cs231n course.

def eval_numerical_gradient_inputs(layer, inputs, in_grads, h=1e-5):
    grads = np.zeros_like(inputs)
    it = np.nditer(inputs, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        oldval = inputs[idx]
        inputs[idx] = oldval + h
        pos = layer.forward(inputs).copy()
        inputs[idx] = oldval - h
        neg = layer.forward(inputs).copy()
        inputs[idx] = oldval

        grads[idx] = np.sum((pos - neg) * in_grads) / (2 * h)
        it.iternext()
    return grads

def eval_numerical_gradient_params(layer, inputs, in_grads, h=1e-5):
    w_grad = np.zeros_like(layer.weights)
    b_grad = np.zeros_like(layer.bias)

    w_it = np.nditer(w_grad, flags=['multi_index'], op_flags=['readwrite'])
    b_it = np.nditer(b_grad, flags=['multi_index'], op_flags=['readwrite'])

    while not w_it.finished:
        idx = w_it.multi_index

        oldval = layer.weights[idx]
        layer.weights[idx] = oldval + h
        pos = layer.forward(inputs).copy()
        layer.weights[idx] = oldval - h
        neg = layer.forward(inputs).copy()
        layer.weights[idx] = oldval

        w_grad[idx] = np.sum((pos - neg) * in_grads) / (2 * h)
        w_it.iternext()

    while not b_it.finished:
        idx = b_it.multi_index

        oldval = layer.bias[idx]
        layer.bias[idx] = oldval + h
        pos = layer.forward(inputs).copy()
        layer.bias[idx] = oldval - h
        neg = layer.forward(inputs).copy()
        layer.bias[idx] = oldval

        b_grad[idx] = np.sum((pos - neg) * in_grads) / (2 * h)
        b_it.iternext()
    
    return w_grad, b_grad

def eval_numerical_gradient_loss(loss, inputs, targets, h=1e-5):
    grads = np.zeros_like(inputs)
    it = np.nditer(inputs, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index

        oldval = inputs[idx]
        inputs[idx] = oldval + h
        pos = loss.forward(inputs, targets)[0].copy()
        inputs[idx] = oldval - h
        neg = loss.forward(inputs, targets)[0].copy()
        inputs[idx] = oldval

        grads[idx] = np.sum((pos - neg)) / (2 * h)
        it.iternext()
    return grads

def check_grads(cacul_grads, numer_grads, threshold = 1e-7):
    precise = np.linalg.norm(cacul_grads-numer_grads) / max(np.linalg.norm(cacul_grads), np.linalg.norm(numer_grads))
    return precise

def check_grads_layer(layer, inputs, in_grads):
    numer_grads = eval_numerical_gradient_inputs(layer, inputs, in_grads)
    cacul_grads = layer.backward(in_grads, inputs)

    inputs_result = check_grads(cacul_grads, numer_grads)
    print('<1e-8 will be fine')
    print('Gradients to inputs:', inputs_result)
    if layer.trainable:
        w_grad, b_grad = eval_numerical_gradient_params(layer, inputs, in_grads)
        w_results = check_grads(layer.w_grad, w_grad)
        b_results = check_grads(layer.b_grad, b_grad)
        print('Gradients to weights: ', w_results)
        print('Gradients to bias: ', b_results)

def check_grads_loss(layer, inputs, targets):
    numer_grads = eval_numerical_gradient_loss(layer, inputs, targets)
    cacul_grads = layer.backward(inputs, targets)

    inputs_result = check_grads(cacul_grads, numer_grads)
    print('<1e-8 will be fine')
    print('inputs:', inputs_result)
