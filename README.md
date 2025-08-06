# Scratch Convolutional Neural Net for Multiple Classification 


![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/scratch-conv-net-classification)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/scratch-conv-net-classification)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/scratch-conv-net-classification?style=social)

## Project Description
This repository delivers a complete CNN implementation **from scratch** in pure Python (NumPy + Pillow), built to classify the Plant Seedlings dataset. Every component—2D convolution, ReLU activation, max‑pooling, flattening, dense softmax layer, forward & backward passes, and an Adam optimizer—is hand‑coded without relying on TensorFlow or PyTorch. We provide unit tests and interactive Jupyter notebooks to validate functionality and demonstrate usage. The full model was not trained end‑to‑end (computational cost), but its correctness is guaranteed through rigorous testing.

**This repository serves as an interactive learning platform where you can dive deep into the inner workings of convolutional neural networks and backpropagation without the abstraction of automatic‑differentiation tools. It provides a hands‑on sandbox environment to step through each algorithmic detail and experiment with custom modifications in real time. Whether you aim to reinforce theoretical concepts, validate new ideas, or simply satisfy your curiosity about how CNNs learn, this codebase offers a transparent, extendable foundation for exploration and discovery.


**
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
| **src/layers/activations.py**        | `ReLU`, `LeakyReLU` , `Tanh` activations with backprop        |
| **src/layers/pooling.py**            | `MaxPool2D` and pooling backprop helpers (supports AveregePool)    |
| **src/layers/dense.py**              | `Dense` + softmax layer with Adam optimizer also `Flatten` layer for reshaping tensors     |
| **src/model.py**                     |  `conv_net_forward`, `conv_net_backward`   , `softmaxCost`     |
| **src/train.py**                     | `crear_modelo` , `full_cnn()` training loop with nested tqdm bars              |
| **tests/test_layers.ipynb**          | Notebook tests for each layer (forward+backward)                               |
| **tests/test_data_loader.ipynb**     | Notebook tests for data loader utilities                      |
| **tests/model_showcase.ipynb**       | Notebook demonstrating model build & one training step        |
| **requirements.txt**                 | Dependencies: NumPy, Pillow, tqdm, pytest                     |


## Core Dependencies

```bash
pip install pandas numpy PIL tqdm
```

## Quickstart / Usage example with built-in function 

```python
from train import *

filters = [X_batch.shape[3] , 8 , 16] # Numer of kernels per layer 
pool = [1 , 1 ,1] # 1 for batchNorm else 0 
batch_size = 16
lr = 0.001
epochs = 5
num_clases = 12 
model, history = full_cnn(filters , pool , train_df , 5 , batch_size , lr ,num_clases)
```

## Build a Manual Model 

```python

model = [
  Conv2D(n_C_prev=3, n_C=8,  f=3, stride=1, pad=1),
  ReLU(),
  MaxPool2D(f=2, stride=2),

  Conv2D(n_C_prev=8, n_C=16, f=3, stride=1, pad=1),
  ReLU(),
  MaxPool2D(f=2, stride=2),
  
  Flatten(),
  Dense(n_units=12, initialization='he')] # 12 clases 

```

## Limitations
- Educational focus: **not optimized** for large‑scale or production use.  
- Convolutional loops in pure Python are **slow on CPU**; no GPU acceleration supported.  
- Scaling to larger images or deeper architectures will be prohibitively slow without vectorized/im2col implementations or a C‑backend.  


## Future Work
- **Efficient convolutions**: implement im2col to leverage BLAS and speed up training.  
- **Advanced regularization**: add dropout, batch‑normalization and data augmentation.  
- **Hyperparameter tuning**: integrate learning‑rate schedulers and alternative optimizers.  
- **Expand scope**: support regression tasks, segmentation or object detection.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request. New features or bug fixes should include appropriate tests and adhere to the existing code style.

## License
This project is licensed under the **MIT License**. See the LICENSE file for full terms.
