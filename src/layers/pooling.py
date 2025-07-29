import numpy as np 

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Forward pass for a 2D pooling layer.

    Args:
        A_prev (np.ndarray): Input data of shape (m, n_H_prev, n_W_prev, n_C_prev).
        hparameters (dict): Dictionary with keys:
            - "f" (int): size of the pooling window (f × f).
            - "stride" (int): stride for moving the window.
        mode (str): Pooling mode, either "max" or "average".

    Returns:
        A (np.ndarray): Output of the pooling layer, shape (m, n_H, n_W, n_C_prev).
        cache (tuple): Cached values (A_prev, hparameters) for the backward pass.
    """
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    f = hparameters["f"]
    stride = hparameters["stride"]


    if mode not in ("max", "average"):
        raise ValueError("`mode` debe ser 'max' o 'average'")
    if not isinstance(f, int) or f <= 0:
        raise ValueError("`f` debe ser un entero > 0")
    if not isinstance(stride, int) or stride <= 0:
        raise ValueError("`stride` debe ser un entero > 0")
    # Asegurarnos de que (n_H_prev - f) es divisible por stride
    if (n_H_prev - f) % stride != 0 or (n_W_prev - f) % stride != 0:
        raise ValueError("Dimensiones inválidas: comprueba f y stride.")

    
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):             
        for h in range(n_H):       
            vert_start = h * stride
            vert_end   = vert_start + f
            
            for w in range(n_W):    
                horiz_start = w * stride
                horiz_end   = horiz_start + f
                
                for c in range(n_C): 
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)      
    
    cache = (A_prev, hparameters)

    
    return A, cache



def create_mask_from_window(x):
    """
    Create a boolean mask identifying the maximum entry in a 2D window.

    Args:
        x (np.ndarray): 2D array of shape (f, f).

    Returns:
        np.ndarray: Boolean mask of the same shape as x, with True at the position(s)
                    of the maximum value in x.
    """  
    mask = (x == np.max(x))
    return mask


def distribute_value(dz, shape):
    """
    Evenly distribute a scalar value over a matrix of specified shape.

    Args:
        dz (float): Scalar value to distribute.
        shape (tuple of int): Tuple (n_H, n_W) specifying the output matrix dimensions.

    Returns:
        np.ndarray: Array of shape (n_H, n_W) where each element equals dz / (n_H * n_W).
    """ 
    (n_H, n_W) = shape
    average = dz / (n_H * n_W)

    a = np.ones((n_H, n_W)) * average
    
    return a


def pool_backward(dA, cache, mode = "max"):
    """
    Perform the backward pass for a 2D pooling layer.

    Args:
        dA (np.ndarray): Gradient of the cost with respect to the output of the pooling layer,
                         of shape (m, n_H, n_W, n_C).
        cache (tuple): Tuple containing:
            - A_prev (np.ndarray): Input data to the pooling layer during forward pass,
                                    shape (m, n_H_prev, n_W_prev, n_C_prev).
            - hparameters (dict): Dictionary with keys:
                'stride' (int): Stride used in pooling,
                'f' (int): Size of the pooling window.
        mode (str): Pooling mode, either 'max' or 'average'.

    Returns:
        np.ndarray: Gradient of the cost with respect to the input of the pooling layer,
                    of shape (m, n_H_prev, n_W_prev, n_C_prev).
    """

    A_prev, hparameters = cache
    stride = hparameters["stride"]
    f = hparameters["f"]
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (_, n_H, n_W, n_C) = dA.shape
    
    dA_prev = np.zeros_like(A_prev)
    
    for i in range(m):

        a_prev = A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
 
                    vert_start  = h * stride
                    vert_end    = vert_start + f
                    horiz_start = w * stride
                    horiz_end   = horiz_start + f

                    if mode == "max":
 
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        mask = create_mask_from_window(a_prev_slice)

                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += mask * dA[i, h, w, c]

                    elif mode == "average":
                        a = distribute_value(dA[i, h, w, c], (f, f))
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += a

    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev


class MaxPool2D:
    """
    2D max pooling layer.

    Performs non-overlapping max pooling over input feature maps.
    """

    def __init__(self, f: int, stride: int = 2):
        """
        Initialize the MaxPool2D layer.

        Args:
            f (int): Size of the pooling window (f × f).
            stride (int): Stride (step) size for both height and width.
        Raises:
            ValueError: If f or stride is not a positive integer.
        """

        if not isinstance(f, int) or f <= 0:
            raise ValueError("`f` debe ser un entero > 0.")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("`stride` debe ser un entero > 0.")

        self.f = f
        self.stride = stride
        self.cache = None
        self.name = 'poollayer'

    def forward(self, A_prev: np.ndarray):
        """
        Forward pass for max pooling.

        Args:
            A_prev (np.ndarray): Input data of shape (m, H_prev, W_prev, C_prev).

        Returns:
            np.ndarray: Pooled output of shape (m, H, W, C_prev),
                        where H and W depend on f and stride.
        """

        hparams = {'f': self.f, 'stride': self.stride}
        A, cache = pool_forward(A_prev, hparams, mode='max')
        self.cache = cache
        return A
    
    def backward(self, dA):
        """
        Backward pass for max pooling.

        Args:
            dA (np.ndarray): Gradient of the loss with respect to the pooled output,
                             of shape (m, H, W, C_prev).

        Returns:
            np.ndarray: Gradient with respect to the input A_prev,
                        of shape (m, H_prev, W_prev, C_prev).
        """

        A_prev, hparams = self.cache
        dA_prev = pool_backward(dA, (A_prev, hparams), mode='max')
        return dA_prev

