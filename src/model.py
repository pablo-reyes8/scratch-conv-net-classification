import numpy as np 


def compute_cost(A_final , labels , tipe ='CrossEntropy' ):
    """Compute the loss between predicted probabilities and true labels, with optional L2 regularization.

    Supports binary cross-entropy for two-class problems or categorical
    cross-entropy for multi-class problems. 

    Args:
        A_final (np.ndarray): Predicted probabilities, shape (n_y, m).
        labels (np.ndarray): True labels, shape (n_y, m) or (m,); will be
            reshaped to (1, m) if necessary.
        tipe (str): Type of cost to compute:
            - 'BinaryCrossEntropy': binary cross-entropy loss.
            - 'CrossEntropy': categorical cross-entropy loss.
        caches (dict, optional): Dictionary of cached values from forward pass.
            Used to extract weight matrices 'W1', 'W2', … when `regularization=True`.

    Returns:
        float: The scalar loss value (including regularization term if enabled).

    Raises:
        ValueError: If `tipe` is not one of the supported cost types.
    """

    # Ensure labels are shape (n_y, m)
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)

    m = labels.shape[1]
    eps = 1e-15
    A_safe = np.clip(A_final, eps, 1 - eps)

    # Compute base cost
    if tipe == 'BinaryCrossEntropy':
        logprobs = (labels * np.log(A_safe) +
                    (1 - labels) * np.log(1 - A_safe))
        cost = - (1 / m) * np.sum(logprobs)

    elif tipe == 'CrossEntropy':
        logprobs = labels * np.log(A_safe)
        cost = - (1 / m) * np.sum(logprobs)

    else:
        raise ValueError("`tipe` must be 'BinaryCrossEntropy' or 'CrossEntropy'")

    return float(np.squeeze(cost))



def conv_net_forward(layers, X):
    """
    Perform forward propagation through a sequence of convolutional network layers.

    Args:
        layers (list): Ordered list of layer objects with .name and .forward():
            - 'conv2d': convolutional layer, caches (A_prev, W, b, hparams)
            - 'poollayer': max‑pooling layer, caches (A_prev, hparams)
            - 'relu': ReLU activation, caches Z
            - 'dense': fully‑connected + softmax, caches (A_prev_flat, A)
        X (np.ndarray): Input batch of shape (m, H, W, C) for conv layers.

    Returns:
        A (np.ndarray): Output activations from the final layer.
        caches (dict): Mapping string keys to cached arrays needed for backprop.
    """
    caches = {}
    A = X
    conv, pool, activation, dense = 1, 1, 1, 1

    for idx, layer in enumerate(layers):
        A_prev = A

        if layer.name == 'dense':
            m = A_prev.shape[0]
            A_prev_flat = A_prev.reshape(m, -1)
            A = layer.forward(A_prev_flat)

            A_prev_c, A_c = layer.cache
            caches['A_prev' + str(dense) + ' dense'] = A_prev_c
            caches['A' + str(dense) + ' dense'] = A_c
            dense += 1

        else:
            A = layer.forward(A_prev)

            if layer.name == 'conv2d':
                A_prev_c, W_c, b_c, hp_c = layer.cache
                caches['A_prev' + str(conv) + ' conv2d'] = A_prev_c
                caches['W'+ str(conv) + ' conv2d'] = W_c
                caches['b'+ str(conv) + ' conv2d'] = b_c
                caches['hparameters' + str(conv) + ' conv2d'] = hp_c
                conv += 1

            elif layer.name == 'poollayer':
                A_prev_c, hp_c = layer.cache
                caches['A_prev'+ str(pool) + ' PoolLayer'] = A_prev_c
                caches['hparameters' + str(pool) + ' PoolLayer'] = hp_c
                pool += 1

            elif layer.name == 'relu':
                Z_c = layer.cache
                caches['Z' + str(activation) + ' ReLu'] = Z_c
                activation += 1

    return A, caches



def conv_net_backward(layers , A_out , y_batch , lr):
    """
    Perform the backward pass through a convolutional network and update trainable parameters.

    Args:
        layers (list): List of layer objects in forward order.
        A_out (np.ndarray): Softmax output from the network, shape (m, n_classes).
        y_batch (np.ndarray): One-hot encoded true labels, shape (m, n_classes).
        lr (float): Learning rate for Adam updates on Conv2D and Dense layers.

    Returns:
        dict: Gradients for each layer, keyed by layer index and type, useful for debugging.
    """
    grads = {}
    dA_l = A_out - y_batch
    conv, pool, activation, dense = 1, 1, 1, 1

    for layer in reversed(layers):
        dA_post = dA_l

        if layer.name == 'dense':
            dA_l = layer.backward(dA_post)

            grads['dW' + str(dense) + ' dense'] = layer.dW
            grads['db' + str(dense) + ' dense'] = layer.db 
            layer.update_adam(lr)
            dense += 1

        elif layer.name == 'flatten':
            dA_l = layer.backward(dA_post)

            grads['dA' + str(dense) + ' flatten'] = dA_l
        
        elif layer.name == 'poollayer':
            dA_l = layer.backward(dA_post)

            grads['dA' + str(pool) + ' PoolLayer'] = dA_l
            pool += 1

        elif layer.name == 'relu':
            dA_l = layer.backward(dA_post)

            grads['dZ' + str(activation) + ' ReLu'] = dA_l
            activation += 1

        elif layer.name == 'conv2d':
            dA_l = layer.backward(dA_post)

            
            grads['dW' + str(conv) + ' conv2d'] = layer.dW
            grads['db' + str(conv) + ' conv2d'] = layer.db 
            layer.update_adam(lr)
            conv += 1

        else:
            raise ValueError(f"Capa desconocida: {layer.name}") 
    
    return grads


