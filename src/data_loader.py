
import numpy as np
from PIL import Image


def load_image(path: str, target_size: tuple = (64, 64)):
    """
    Load an image from disk, resize it, and normalize pixel values.

    Args:
        path (str): Filesystem path to the image.
        target_size (tuple of int): Desired output size as (width, height).

    Returns:
        np.ndarray: RGB image array of shape (height, width, 3), dtype float32,
                    with values scaled to [0.0, 1.0].
    """
    img = Image.open(path).convert('RGB')
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    img = img.resize(target_size, resample=resample)
    return np.array(img, dtype=np.float32) / 255.0


def encode_labels(label_batch: list, label2idx: dict, label_list: list):
    """
    Convert a list of label strings to one-hot encoded vectors.

    Args:
        label_batch (list of str): Labels for the current batch.
        label2idx (dict): Mapping from label string to integer index.
        label_list (list of str): Full list of possible labels in order.

    Returns:
        np.ndarray: One-hot matrix of shape (batch_size, num_classes), dtype float32.
    """
    idxs = [label2idx[label] for label in label_batch]
    one_hot = np.zeros((len(idxs), len(label_list)), dtype=np.float32)
    one_hot[np.arange(len(idxs)), idxs] = 1.0
    return one_hot


def batch_generator(df, label2idx: dict, label_list: list , batch_size: int = 32, shuffle: bool = True, target_size: tuple = (64, 64)):
    """
    Infinite generator yielding batches of images and one-hot labels.

    Args:
        df (pandas.DataFrame): Must contain columns 'filepath' and 'label'.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle df at the start of each epoch.
        target_size (tuple of int): Size to resize images (width, height).

    Yields:
        Tuple[np.ndarray, np.ndarray]:
            - X: Array of shape (batch_size, H, W, 3), dtype float32.
            - y: One-hot labels of shape (batch_size, num_classes), dtype float32.
    """
    n = len(df)
    while True:
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        for i in range(0, n, batch_size):
            batch = df.iloc[i:i + batch_size]
            X = np.stack([load_image(fp, target_size) for fp in batch['filepath']])
            y = encode_labels(batch['label'].tolist(), label2idx, label_list)
            yield X, y

