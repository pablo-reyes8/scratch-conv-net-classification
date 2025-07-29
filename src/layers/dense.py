
import numpy as np 

class Flatten:
    """
    Flatten layer that reshapes its input into a 2D array
    (batch_size, features).

    Attributes:
        name (str): Identifier for this layer.
        cache (tuple): Stores the original input shape for backward pass.
    """

    def __init__(self):
        """
        Initialize the Flatten layer.
        """

        self.name = 'flatten'
        self.cache = None

    def forward(self, A_prev):
        """
        Forward pass: flatten the input.

        Args:
            A_prev (np.ndarray): Input array of shape (m, ...),
                                 where m is the batch size and ... 
                                 represents any number of additional dimensions.

        Returns:
            np.ndarray: Flattened output of shape (m, features),
                        where features = product of the ... dimensions.
        """


        self.cache = A_prev.shape
        m = A_prev.shape[0]
        return A_prev.reshape(m, -1)

    def backward(self, dA):
        """
        Backward pass: reshape upstream gradients to the original input shape.

        Args:
            dA (np.ndarray): Gradient of the loss w.r.t. the flattened output,
                             of shape (m, features).

        Returns:
            np.ndarray: Gradient reshaped to the original input shape stored in cache.
        """"""
        Backward pass: reshape upstream gradients to the original input shape.

        Args:
            dA (np.ndarray): Gradient of the loss w.r.t. the flattened output,
                             of shape (m, features).

        Returns:
            np.ndarray: Gradient reshaped to the original input shape stored in cache.
        """

        return dA.reshape(self.cache)
    

class Dense:
    """
    Fully‑connected (dense) layer with softmax activation and Adam optimizer support.
    """

    def __init__(self, n_units, initialization='rand', scale=0.01, seed=None):
        """
        Initialize the Dense layer (weights and Adam buffers will be set on first forward).

        Args:
            n_units (int): Number of output neurons.
            initialization (str): 'he' for He initialization or 'rand' for scaled random.
            scale (float): Scale for 'rand' initialization.
            seed (int or None): Random seed for reproducibility.
        """

        self.n_inputs = None
        self.n_units = n_units
        self.initialization = initialization
        self.scale = scale
        self.seed = seed
        
        self.W = None
        self.b = None
        

        self.dW = None
        self.db = None
        
        self.mW = None
        self.vW = None
        self.mb = None
        self.vb = None
        self.t = 0
        
        self.cache = None
        self.name = 'dense'

    def forward(self, A_prev):
        """
        Forward pass: linear transform followed by softmax.

        Args:
            A_prev (np.ndarray): Input data of shape (m, features).

        Returns:
            np.ndarray: Output probabilities of shape (m, n_units).
        """

        m, features = A_prev.shape

        if self.W is None:
            if self.seed is not None:
                np.random.seed(self.seed)
            self.n_inputs = features

            if self.initialization.lower() == 'he':
                factor = np.sqrt(2.0 / self.n_inputs)
                self.W = np.random.randn(self.n_inputs, self.n_units) * factor
            else:
                self.W = np.random.randn(self.n_inputs, self.n_units) * self.scale
            self.b = np.zeros((1, self.n_units), dtype=self.W.dtype)

            self.mW = np.zeros_like(self.W)
            self.vW = np.zeros_like(self.W)
            self.mb = np.zeros_like(self.b)
            self.vb = np.zeros_like(self.b)

        Z = A_prev @ self.W + self.b
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        A = exps / np.sum(exps, axis=1, keepdims=True)
        self.cache = (A_prev, A)
        return A

    def backward(self, dA):
        """
        Backward pass: compute gradients for weights, biases, and inputs.

        Args:
            dA (np.ndarray): Upstream gradient of shape (m, n_units).

        Returns:
            np.ndarray: Gradient w.r.t. input A_prev, shape (m, n_inputs).
        """

        A_prev, A = self.cache
        m = A_prev.shape[0]

        S = dA * A
        sum_S = np.sum(S, axis=1, keepdims=True)
        dZ = S - A * sum_S

        self.dW = (A_prev.T @ dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ @ self.W.T
        return dA_prev

    def update_adam(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Update parameters W and b using Adam optimizer.

        Must be called after backward().

        Args:
            lr (float): Learning rate.
            beta1 (float): Exponential decay rate for first moment.
            beta2 (float): Exponential decay rate for second moment.
            eps (float): Small epsilon to avoid division by zero.
        """

        self.t += 1

        # Update biased first moment estimates
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.mb = beta1 * self.mb + (1 - beta1) * self.db

        # Update biased second moment estimates
        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW ** 2)
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)

        # Compute bias-corrected moments
        mW_hat = self.mW / (1 - beta1 ** self.t)
        mb_hat = self.mb / (1 - beta1 ** self.t)
        vW_hat = self.vW / (1 - beta2 ** self.t)
        vb_hat = self.vb / (1 - beta2 ** self.t)

        # Parameter update
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)


