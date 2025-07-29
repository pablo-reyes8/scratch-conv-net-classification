import numpy as np

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """

    if not isinstance(pad, int) or pad < 0:
        raise ValueError("`pad` debe ser un entero >= 0.")
    if X.ndim != 4:
        raise ValueError("`X` debe tener forma (m, n_H, n_W, n_C).")
    
    m, n_H, n_W, n_C = X.shape
    X_pad = np.zeros((m, n_H + 2*pad,n_W + 2*pad,n_C) , dtype=X.dtype)
    X_pad[:, pad:pad+n_H, pad:pad+n_W, :] = X
    
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Perform a single convolution step on a slice of the input.

    Args:
        a_slice_prev (np.ndarray): Input slice of shape (f, f, n_C_prev).
        W (np.ndarray): Filter weights of shape (f, f, n_C_prev).
        b (np.ndarray or float): Bias term, broadcastable to a scalar.

    Returns:
        float: The result of applying the filter and bias (i.e., sum(a_slice_prev * W) + b).
    """

    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + b.item()
    return Z


def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution layer
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    if not isinstance(hparameters, dict):
        raise ValueError("`hparameters` debe ser un dict con 'stride' y 'pad'.")
    if 'stride' not in hparameters or 'pad' not in hparameters:
        raise KeyError("`hparameters` requiere las claves 'stride' y 'pad'.")
    stride, pad = hparameters['stride'], hparameters['pad']
    if not (isinstance(stride, int) and stride > 0):
        raise ValueError("`stride` debe ser un entero > 0.")
    if not (isinstance(pad, int) and pad >= 0):
        raise ValueError("`pad` debe ser un entero >= 0.")

    n_H = int((n_H_prev - f + 2*pad) / stride) + 1
    n_W = int((n_W_prev - f + 2*pad) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):       # para cada imagen
        a_prev_pad = A_prev_pad[i] # shape (n_H_prev+2pad, n_W_prev+2pad, n_C_prev)

        for h in range(n_H): # recorre ejes verticales
            vert_start = h * stride
            vert_end   = vert_start + f 
            for w in range(n_W):                       # recorre ejes horizontales
                horiz_start = w * stride
                horiz_end   = horiz_start + f
                for c in range(n_C):                   # recorre cada filtro / canal de salida
                    a_slice_prev = a_prev_pad[
                        vert_start:vert_end,
                        horiz_start:horiz_end,:]     # shape (f, f, n_C_prev)
                        
                    Z[i, h, w, c] = conv_single_step(a_slice_prev,W[:, :, :, c],b[:, :, :, c])
                    
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache


def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution layer
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """    
    A_prev, W, b, hparameters = cache
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C  = W.shape
    _, n_H, n_W, _= dZ.shape
    
    dA_prev = np.zeros_like(A_prev , dtype=A_prev.dtype)     
    dW  = np.zeros_like(W , dtype=W.dtype)     
    db  = np.zeros_like(b , dtype=b.dtype)      
    
    A_prev_pad   = zero_pad(A_prev, pad)
    dA_prev_pad  = zero_pad(dA_prev, pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i]  # (n_H_prev+2pad, n_W_prev+2pad, n_C_prev)
        da_prev_pad = dA_prev_pad[i] # igual shape que a_prev_pad
        
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    # Encontrar coordenadas del slice
                    vert_start  = h * stride
                    vert_end    = vert_start + f
                    horiz_start = w * stride
                    horiz_end   = horiz_start + f
        
                    # Extraer slice de A_prev_pad
                    a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:] 
                    
                    # dA_prev_pad: distribuye dZ * W sobre la ventana correspondiente
                    da_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:] += W[:, :, :, c] * dZ[i, h, w, c]
                    
                    # dW: gradiente del filtro c es sum(a_slice * dZ)
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    
                    # db: gradiente del bias c es la suma de dZ sobre todos los ejemplos y posiciones
                    db[:, :, :, c] += dZ[i, h, w, c]
                    
        if pad != 0:
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        else:
            A_prev[i, :, :, :] = da_prev_pad     

    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db


class Conv2D:
    """
    2D convolutional layer with built‑in Adam optimizer support.
    """
    def __init__(self, n_C_prev, n_C,
        f, stride = 1, pad = 0, initialization = 'he',scale = 0.01, seed = None):

        """
        Initialize a Conv2D layer.

        Args:
            n_C_prev (int): Number of channels in the input (depth of A_prev).
            n_C (int): Number of filters (output channels).
            f (int): Size of each filter (filters are f x f).
            stride (int, optional): Stride length for the convolution. Defaults to 1.
            pad (int, optional): Number of zero-padding pixels around the input. Defaults to 0.
            initialization (str, optional): Weight init method: 'he' or 'rand'. Defaults to 'he'.
            scale (float, optional): Scaling factor for 'rand' init. Defaults to 0.01.
            seed (int or None, optional): Random seed for reproducibility. Defaults to None.

        Attributes:
            W (np.ndarray): Filters of shape (f, f, n_C_prev, n_C).
            b (np.ndarray): Biases of shape (1, 1, 1, n_C).
            stride (int): Convolution stride.
            pad (int): Padding size.
            cache (tuple): Cached values for backward pass.
            dW (np.ndarray): Gradient of W.
            db (np.ndarray): Gradient of b.
            mW, vW, mb, vb (np.ndarray): Adam first/second moment buffers.
            t (int): Adam timestep counter.
        """
         
        if seed is not None:
            np.random.seed(seed)

        self.n_C_prev = n_C_prev
        self.n_C      = n_C
        self.f        = f
        self.stride   = stride
        self.pad      = pad
        self.name     = 'conv2d'

        if initialization.lower() == 'he':
            factor = np.sqrt(2.0 / (f * f * n_C_prev))
            self.W = np.random.randn(f, f, n_C_prev, n_C) * factor
        elif initialization.lower() == 'rand':
            self.W = np.random.randn(f, f, n_C_prev, n_C) * scale
        else:
            raise ValueError("`initialization` debe ser 'rand' o 'he'")
        self.b = np.zeros((1, 1, 1, n_C), dtype=self.W.dtype)


        self.cache = None
        self.dW = None
        self.db = None

        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t  = 0

    def forward(self, A_prev):
        """
        Perform the forward pass of the convolution.

        Args:
            A_prev (np.ndarray): Input data of shape (m, n_H_prev, n_W_prev, n_C_prev).

        Returns:
            np.ndarray: Convolved output Z of shape (m, n_H, n_W, n_C).
        """
        hparams = {'stride': self.stride, 'pad': self.pad}
        Z, cache = conv_forward(A_prev, self.W, self.b, hparams)
        self.cache = cache
        return Z

    def backward(self, dZ):
        """
        Perform the backward pass of the convolution.

        Args:
            dZ (np.ndarray): Gradient of the loss with respect to the output Z.

        Returns:
            np.ndarray: Gradient with respect to the input A_prev.
        """
        dA_prev, dW, db = conv_backward(dZ, self.cache)
        self.dW, self.db = dW, db
        return dA_prev

    def update_adam(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Update parameters using Adam optimization. Call after backward().

        Args:
            lr (float): Learning rate.
            beta1 (float, optional): Exponential decay rate for the first moment. Defaults to 0.9.
            beta2 (float, optional): Exponential decay rate for the second moment. Defaults to 0.999.
            eps (float, optional): Small constant to prevent division by zero. Defaults to 1e-8.
        """
        self.t += 1

        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.mb = beta1 * self.mb + (1 - beta1) * self.db


        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW ** 2)
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)

        mW_hat = self.mW / (1 - beta1 ** self.t)
        mb_hat = self.mb / (1 - beta1 ** self.t)
        vW_hat = self.vW / (1 - beta2 ** self.t)
        vb_hat = self.vb / (1 - beta2 ** self.t)

        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
    


