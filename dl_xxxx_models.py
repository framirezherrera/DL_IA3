# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: dl_tfmr_models.py
# Descripción: Funciones compartidas para el entrenamiento de los modelos.
# Implementado por: Felipe Ramírez Herrera 
# Curso Aprendizaje Profundo 1 y 2. Universidad de Valencia / ADEIT
# Ultima revisión: 11/04/2024 
# ----------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


import dl_common

#   Pone a dormir la GPU durante un cierto número de minutos.
#   Parámetros:
#        - minutes: El número de minutos que la GPU debe dormir.

def let_gpu_rest(minutes):
    if minutes > 0:
        time.sleep(minutes * 60)    

# Inicializa los parámetros de un modelo Transformer.
#    Parámetros:
#    - model: El modelo Transformer a inicializar.

def init_transformer_model(model : nn.Module):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

# Calcula la precisión de predicción enmascarada.
#    Parámetros:
#    - label: Etiquetas verdaderas.
#    - pred: Predicciones del modelo.
#    - tgt_pad_idx: Índice de relleno para las etiquetas de destino.

#    Retorna:
#    - Precisión de predicción enmascarada.

def masked_accuracy(label, pred, tgt_pad_idx = 0):
  pred = torch.argmax(pred, dim=-1)
  match = label.eq(pred)
  mask = label.ne(tgt_pad_idx)
  match = match & mask
  match = match.type(torch.float32) 
  mask =  mask.type(torch.float32)
  return torch.sum(match)/torch.sum(mask)

# Calcula el tamaño en bytes de un tensor de PyTorch.
def getSizeOf(a : torch.Tensor):
    return sys.getsizeof(a) + torch.numel(a) * a.element_size()

# Calcula el número de elementos en un tensor de PyTorch.
def ElementsOf(a : torch.Tensor):
    return torch.numel(a)


def count_params(model, return_int=False):
    params = sum([torch.prod(torch.tensor(x.shape)).item() for x in model.parameters() if x.requires_grad])
    if return_int:
        return params
    else:
        print("Model '{0}' has {1} trainable parameters.".format(model.getName(), params))




global DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"]= "0" # 0
os.environ["CUDA_LAUNCH_BLOCKING"]= "1" # 1

if torch.cuda.is_available():
    print("Computing device = {0}".format(DEVICE)) 


# -----------------------------------------------------------------------------------------------------------
# Ploting de los datos
# -----------------------------------------------------------------------------------------------------------

PLOT_X = 6
PLOT_Y = 3 
PLOT_LW = 1.5    

plt.rc('font', size=8)
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('figure', titlesize=8)

def plot_accuracy_curve(name, history):
    fig, ax = plt.subplots(figsize=(PLOT_X, PLOT_Y), layout='constrained')
    ax.set_title('Exactitud del modelo {0}'.format(name))
    ax.plot(history['train_accm'], label='Entrenamiento', linestyle='solid', lw=PLOT_LW)
    ax.plot(history['valid_accm'], label='Validación', linestyle='solid', lw=PLOT_LW)
    ax.set_xticks(np.arange(1, dl_common.NUMBER_OF_EPOCHS + 1, 1))
    ax.set_yticks(np.arange(0, 1, 1 / 10))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Exactitud')
    ax.legend()
    plt.savefig("{0}_acc_curve.svg".format(name))

def plot_loss_curve(name, history):
    fig, ax = plt.subplots(figsize=(PLOT_X, PLOT_Y), layout='constrained')
    ax.set_title('Pérdida del modelo {0}'.format(name))
    ax.plot(history['train_loss'], label='Entrenamiento', linestyle='solid', lw=PLOT_LW)
    ax.plot(history['valid_loss'], label='Validación', linestyle='solid', lw=PLOT_LW)
    ax.set_xticks(np.arange(1, dl_common.NUMBER_OF_EPOCHS + 1, 1))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pérdida')
    ax.legend()
    plt.savefig("{0}_loss_curve.svg".format(name))

def plot_pplx_curve(name, history):
    fig, ax = plt.subplots(figsize=(PLOT_X, PLOT_Y), layout='constrained')
    ax.set_title('Perplejidad del modelo {0}'.format(name))
    ax.plot(history['train_pplx'], label='Entrenamiento', linestyle='solid', lw=PLOT_LW)
    ax.plot(history['valid_pplx'], label='Validación', linestyle='solid', lw=PLOT_LW)
    ax.set_xticks(np.arange(1, dl_common.NUMBER_OF_EPOCHS + 1, 1))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplejidad')
    ax.legend()
    plt.savefig("{0}_pplx_curve.svg".format(name))


def plot_accuracy_curve_dual_transformer(name, history):
    fig, ax = plt.subplots(figsize=(PLOT_X, PLOT_Y), layout='constrained')
    ax.set_title('Exactitud del modelo {0}'.format(name))
    ax.plot(history['train_joint_accm'], label='Entrenamiento', linestyle='solid', lw=PLOT_LW)
    ax.plot(history['valid_joint_accm'], label='Validación', linestyle='solid', lw=PLOT_LW)
    ax.set_xticks(np.arange(1, dl_common.NUMBER_OF_EPOCHS + 1, 1))
    ax.set_yticks(np.arange(0, 1, 1 / 10))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Exactitud')
    ax.legend()
    plt.savefig("{0}_joint_accuracy_curves.svg".format(name))


def plot_loss_curve_dual_transformer(name, history):
    fig, ax = plt.subplots(figsize=(PLOT_X, PLOT_Y), layout='constrained')
    ax.set_title('Pérdida del modelo {0}'.format(name))
    ax.plot(history['train_joint_loss'], label='Entrenamiento', linestyle='solid', lw=PLOT_LW)
    ax.plot(history['valid_joint_loss'], label='Validación', linestyle='solid', lw=PLOT_LW)
    ax.set_xticks(np.arange(1, dl_common.NUMBER_OF_EPOCHS + 1, 1))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pérdida')
    ax.legend()
    plt.savefig("{0}_joint_loss_curves.svg".format(name))

def plot_accuracy_curve_dual_transformer_both(name, history):
    fig, ax = plt.subplots(figsize=(PLOT_X, PLOT_Y), layout='constrained')
    ax.set_title('Exactitud del modelo {0}'.format(name))
    ax.plot(history['train_acc_a'], label='Entrenamiento (EN-ES)')
    ax.plot(history['train_acc_b'], label='Entrenamiento (EN-FR)')
    ax.plot(history['valid_acc_a'], label='Validación (EN-ES)')
    ax.plot(history['valid_acc_b'], label='Validación (EN-FR)')
    ax.set_xticks(np.arange(1, dl_common.NUMBER_OF_EPOCHS + 1, 1))
    ax.set_yticks(np.arange(0, 1, 1 / 10))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Exactitud')
    ax.legend()
    plt.savefig("{0}_accuracy_curves.svg".format(name))

def plot_loss_curve_dual_transformer_both(name, history):
    fig, ax = plt.subplots(figsize=(PLOT_X, PLOT_Y), layout='constrained')
    ax.set_title('Pérdida del modelo {0}'.format(name))
    ax.plot(history['train_loss_a'], label='Entrenamiento (EN-ES)')
    ax.plot(history['train_loss_b'], label='Entrenamiento (EN-FR)')
    ax.plot(history['valid_loss_a'], label='Validación (EN-ES)')
    ax.plot(history['valid_loss_b'], label='Validación (EN-FR)')
    ax.set_xticks(np.arange(1, dl_common.NUMBER_OF_EPOCHS + 1, 1))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Pérdida')
    ax.legend()
    plt.savefig("{0}_joint_loss_curves.svg".format(name))
