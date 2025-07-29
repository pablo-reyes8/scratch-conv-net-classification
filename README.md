# Scratch Convolutional Neural Net for Multiple Classification 

## Project Description
This repository delivers a complete CNN implementation **from scratch** in pure Python (NumPy + Pillow), built to classify the Plant Seedlings dataset. Every component—2D convolution, ReLU activation, max‑pooling, flattening, dense softmax layer, forward & backward passes, and an Adam optimizer—is hand‑coded without relying on TensorFlow or PyTorch. We provide unit tests and interactive Jupyter notebooks to validate functionality and demonstrate usage. The full model was not trained end‑to‑end (computational cost), but its correctness is guaranteed through rigorous testing.

## Key Findings
- **Layer‑by‑layer transparency**: each layer encapsulates its own forward, backward and parameter‑update logic.  
- **Adam optimizer integration**: per‑layer first & second moment tracking and bias correction built in.  
- **Test‑driven development**: unit tests verify shapes, values and backprop behavior on trivial inputs.  
- **Interactive training loop**: progress bars (tqdm) display per‑batch cost & accuracy and per‑epoch summaries.  
- **Notebook showcase**: end‑to‑end demonstration of data loading, model assembly and a single training step.

## Repository Structure

| Path                                 | Description                                                   |
|--------------------------------------|---------------------------------------------------------------|
| **data/**                            | Raw Plant Seedlings dataset (`train/`, `test/`)               |
| **full_showcase/**                   | Jupyter notebook (`cnn_full.ipynb`) with a step‑by‑step demo  |
| **src/data_loader.py**               | Image I/O, preprocessing, batch generator                     |
| **src/layers/conv2d.py**            | `Conv2D` layer (forward, backward, Adam)                      |
| **src/layers/activations.py**        | `ReLU` activation with backprop                                |
| **src/layers/pooling.py**            | `MaxPool2D` and pooling backprop helpers                      |
| **src/layers/flatten.py**            | `Flatten` layer for reshaping tensors                         |
| **src/layers/dense.py**              | `Dense` + softmax layer with Adam optimizer                   |
| **src/model.py**                     | `crear_modelo`, `conv_net_forward`, `conv_net_backward`       |
| **src/train.py**                     | `full_cnn()` training loop with nested tqdm bars              |
| **tests/test_layers.ipynb**          | Notebook tests for each layer                                 |
| **tests/test_data_loader.ipynb**     | Notebook tests for data loader utilities                      |
| **tests/model_showcase.ipynb**       | Notebook demonstrating model build & one training step        |
| **requirements.txt**                 | Dependencies: NumPy, Pillow, tqdm, pytest                     |

## Future Work
- **Efficient convolutions**: implement im2col to leverage BLAS and speed up training.  
- **Advanced regularization**: add dropout, batch‑normalization and data augmentation.  
- **Deeper architectures**: experiment with residual blocks or attention mechanisms.  
- **GPU acceleration**: port core ops to JAX or custom CUDA kernels.  
- **Hyperparameter tuning**: integrate learning‑rate schedulers and alternative optimizers.  
- **Expand scope**: support regression tasks, segmentation or object detection.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request. New features or bug fixes should include appropriate tests and adhere to the existing code style.

## License
This project is licensed under the **MIT License**. See the LICENSE file for full terms.
