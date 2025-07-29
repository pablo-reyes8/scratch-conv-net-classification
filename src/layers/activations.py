import numpy as np 


class ReLU:
    """
    Rectified Linear Unit (ReLU) activation layer.

    Attributes:
        name (str): identifier for this activation.
        cache (np.ndarray): stores input Z for use in backward pass.
    """

    def __init__(self):
        """
        Initialize the ReLU layer.
        """
        self.name = 'relu'
        self.cache = None

    def forward(self, Z):
        """
        Forward pass of ReLU.

        Args:
            Z (np.ndarray): pre-activation input of any shape.

        Returns:
            np.ndarray: activations A, where A = max(0, Z).
        """
        A = np.maximum(0, Z)
        self.cache = Z
        return A

    def backward(self, dA):
        """
        Backward pass of ReLU.

        Args:
            dA (np.ndarray): gradient of the loss with respect to the activation output A.

        Returns:
            np.ndarray: gradient of the loss with respect to Z.
        """
        Z = self.cache
        dZ = dA.copy()
        dZ[Z <= 0] = 0
        return dZ


class LeakyReLU:
    """
    Leaky ReLU activation layer.

    Unlike standard ReLU, allows a small, non-zero gradient when Z < 0.

    Attributes:
        name (str): identifier for this activation.
        alpha (float): slope for negative Z values.
        cache (np.ndarray): stores input Z for use in backward pass.
    """

    def __init__(self, alpha=0.01):
        """
        Initialize the LeakyReLU layer.

        Args:
            alpha (float): negative slope coefficient (default 0.01).
        """
        self.name = 'leaky_relu'
        self.alpha = alpha
        self.cache = None

    def forward(self, Z):
        """
        Forward pass of Leaky ReLU.

        Args:
            Z (np.ndarray): pre-activation input of any shape.

        Returns:
            np.ndarray: activations A, where
                        A = Z if Z > 0, and alpha * Z otherwise.
        """
        A = np.where(Z > 0, Z, self.alpha * Z)
        self.cache = Z
        return A

    def backward(self, dA):
        """
        Backward pass of Leaky ReLU.

        Args:
            dA (np.ndarray): gradient of the loss with respect to the activation output A.

        Returns:
            np.ndarray: gradient of the loss with respect to Z.
        """
        Z = self.cache
        dZ = np.where(Z > 0, 1, self.alpha) * dA
        return dZ


class Tanh:
    """
    Hyperbolic tangent (tanh) activation layer.

    Maps input to the range [-1, 1].

    Attributes:
        name (str): identifier for this activation.
        cache (np.ndarray): stores output A = tanh(Z) for use in backward pass.
    """

    def __init__(self):
        """
        Initialize the Tanh layer.
        """
        self.name = 'tanh'
        self.cache = None

    def forward(self, Z):
        """
        Forward pass of tanh.

        Args:
            Z (np.ndarray): pre-activation input of any shape.

        Returns:
            np.ndarray: activations A = tanh(Z).
        """
        A = np.tanh(Z)
        self.cache = A
        return A

    def backward(self, dA):
        """
        Backward pass of tanh.

        Args:
            dA (np.ndarray): gradient of the loss with respect to the activation output A.

        Returns:
            np.ndarray: gradient of the loss with respect to Z,
                        given by dA * (1 - A^2).
        """
        A = self.cache
        dZ = dA * (1 - A ** 2)
        return dZ



