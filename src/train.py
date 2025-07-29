import numpy as np 
from tqdm.auto import trange, tqdm
from data_loader import * 
from layers import *
from model import *
from layers.conv2d      import Conv2D
from layers.activations import ReLU
from layers.pooling     import MaxPool2D
from layers.dense       import *


def crear_modelo(filters, pool, n_classes,
                 filter_size=3, stride=1, pad=1,
                 pool_filter=2, pool_stride=2, poo='half'):
    """
    Build a convolutional neural network given layer specifications.

    Args:
        filters (list of int): Channel counts for each Conv2D layer, e.g. [3, 8, 16].
        pool (list of int): Pool indicators per conv block; if pool[i] != 0, add pooling after block i.
        n_classes (int): Number of output units in the final Dense layer.
        filter_size (int, optional): Height/width of each Conv2D filter. Default is 3.
        stride (int, optional): Stride for each Conv2D layer. Default is 1.
        pad (int, optional): Zero‑padding for each Conv2D layer. Default is 1.
        pool_filter (int, optional): Filter size for MaxPool2D when poo!='half'. Default is 2.
        pool_stride (int, optional): Stride for MaxPool2D when poo!='half'. Default is 2.
        poo (str, optional): Pooling mode. If 'half', uses (2,2) filter+stride; otherwise uses pool_filter and pool_stride. Default is 'half'.

    Returns:
        list: Sequence of layer objects [Conv2D, ReLU, (MaxPool2D)..., Flatten, Dense].
    """
    model = []
    for i in range(len(filters)-1):
        model.append(Conv2D(filters[i], filters[i+1], f=filter_size,
                             stride=stride, pad=pad))
        model.append(ReLU())
        if pool[i] != 0:
            if poo == 'half':
                model.append(MaxPool2D(f=2, stride=2))
            else:
                model.append(MaxPool2D(f=pool_filter, stride=pool_stride))
    model.append(Flatten())
    model.append(Dense(n_units=n_classes, initialization='he'))
    return model



def full_cnn(filters , pool , df_train , epochs ,batch_size , lr , num_clases):
    """
    Train a CNN end‑to‑end on image data using a simple training loop.

    Args:
        filters (list[int]): Number of channels for each Conv2D layer, e.g. [3, 8, 16].
        pool (list[int]): Indicators for adding a pooling layer after each conv block.
        df_train (pd.DataFrame): DataFrame with columns 'filepath' and 'label' for training.
        epochs (int): Number of training epochs (full passes over the dataset).
        batch_size (int): Number of samples per gradient update.
        lr (float): Learning rate for the optimizer.

    Returns:
        tuple:
            model (list): List of layer objects (Conv2D, ReLU, MaxPool2D, Flatten, Dense).
            history (dict): Contains two lists:
                - 'cost': average cross‑entropy per epoch.
                - 'acc':  average accuracy per epoch.
    """
    steps_per_epoch = len(df_train) // batch_size
    gen = batch_generator(df_train, batch_size=batch_size)
    model = crear_modelo(filters, pool , num_clases)

    history = {'cost': [], 'acc': []}
    
    for epoch in range(epochs):
         
        epoch_cost = 0.0
        epoch_acc = 0.0

        pbar = tqdm(range(steps_per_epoch),  desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for _ in pbar:

            X_batch, y_batch = next(gen)
            A_out, _ = conv_net_forward(model, X_batch)
            cost_step = compute_cost(A_out , y_batch , 'CrossEntropy')

            preds = np.argmax(A_out, axis=1)
            trues = np.argmax(y_batch, axis=1)
            acc_step = np.mean(preds == trues)

            epoch_cost += cost_step
            epoch_acc  += acc_step

            conv_net_backward(model , A_out , y_batch , lr)
            
            pbar.set_postfix({
                'cost': f"{cost_step:.4f}",
                'acc':  f"{acc_step:.4f}"})
            
        cost_avg = epoch_cost / steps_per_epoch
        acc_avg  = epoch_acc  / steps_per_epoch
        history['cost'].append(cost_avg)
        history['acc'].append(acc_avg)

        tqdm.write(f"→ Epoch {epoch+1}/{epochs} "f"— cost_avg: {cost_avg:.4f}, acc_avg: {acc_avg:.4f}")

    return model , history

