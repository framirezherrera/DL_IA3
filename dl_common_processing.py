import numpy as np
import pandas as pd
import os
import torch

import torchtext.data
import torchtext.data.datasets_utils
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchtext.vocab import Vocab
import re
from unicodedata import normalize
from typing import List, Tuple, Union
from collections import Counter 

import dl_common as mc
import dl_xxxx_models as mm

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Función generadora de los ejemplos a partir de los tres archivos del Corpus en EN/ES/FR
def generate_examples(src_file, tgt_a_file, tgt_b_file):
    data = []
    with open(src_file, encoding="utf-8") as src_f, open(tgt_a_file, encoding="utf-8") as tgt_a, open(tgt_b_file, encoding="utf-8") as tgt_b:
        for idx, (a, b, c) in enumerate(zip(src_f, tgt_a, tgt_b)):           
            if (a.isspace() | b.isspace() | c.isspace()):
                continue
            data.append({'text_en' : a, 'text_es' : b, 'text_fr' :c} )
    return pd.DataFrame.from_records(data=data)

# Función de limipieza de datos (se puede mejorar y debería considerar apóstrofes)
def clean_text(text : str):
    text = normalize('NFD', str(text).lower())
    text = re.sub('[^A-Za-z ]+', '', text)
    return text

# Agrega padding o trunca las oraciones utilizando los límites de tamaño maximo de secuencia, 
# importante verificar que no se esté destruyendo información
def pad_or_truncate(tokenized_text, allocate : bool = True,  pad_index : int = mc.PAD_EN_IDX, bos_index : int = mc.BOS_EN_IDX, eos_index : int = mc.EOS_EN_IDX):  
    result = []
    if len(tokenized_text) < mc.max_seq_length:
        if (allocate):
            result = [bos_index] + tokenized_text + [eos_index]
            left = mc.max_seq_length - len(result)
            padding = [pad_index] * left
            result = result + padding
        else:
            left = mc.max_seq_length - len(tokenized_text)
            padding = [pad_index] * left
            result = tokenized_text + padding       
    else:
        raise Exception("pad_or_truncate: max_seq_length not computed properly")
    return result


# Importante: Solo cargo los datos al dispositivo (GPU) cuando voy a procesarlos
# es un trade-off entre performance del entrenamiento y capacidad de carga de los
# datos dada las limitaciones de los GPU de consumo (Nvidia RTX):

def tensor_transform(token_ids: List[int], bos_idx, eos_idx): 
    list = [bos_idx] + token_ids + [eos_idx]                         
    return torch.as_tensor(list, device=  mm.DEVICE)

# Función generadora de un batch tripartito
def generate_batch(data_batch):
    en_batch, es_batch, fr_batch = [], [], []
    for (en_item, es_item, fr_item) in data_batch:     
        en_t = tensor_transform(en_item, mc.BOS_EN_IDX, mc.EOS_EN_IDX)
        es_t = tensor_transform(es_item, mc.BOS_ES_IDX, mc.EOS_ES_IDX)
        fr_t = tensor_transform(fr_item, mc.BOS_FR_IDX, mc.EOS_FR_IDX)     
        en_batch.append(en_t)
        es_batch.append(es_t)
        fr_batch.append(fr_t)
    en_batch = pad_sequence(en_batch, padding_value=mc.PAD_EN_IDX, batch_first=False)
    es_batch = pad_sequence(es_batch, padding_value=mc.PAD_ES_IDX, batch_first=False)
    fr_batch = pad_sequence(fr_batch, padding_value=mc.PAD_FR_IDX, batch_first=False)
    return en_batch, es_batch, fr_batch

# Función generadora de un batch bipartito (EN, ES)
def generate_batch_EN_ES(data_batch):
    en_batch, es_batch = [], []
    for (en_item, es_item, _) in data_batch:     
        en_t = tensor_transform(en_item, mc.BOS_EN_IDX, mc.EOS_EN_IDX)
        es_t = tensor_transform(es_item, mc.BOS_ES_IDX, mc.EOS_ES_IDX)
        en_batch.append(en_t)
        es_batch.append(es_t)
    en_batch = pad_sequence(en_batch, padding_value=mc.PAD_EN_IDX, batch_first=False)
    es_batch = pad_sequence(es_batch, padding_value=mc.PAD_ES_IDX, batch_first=False)
    return en_batch, es_batch

# Función generadora de un batch bipartito (EN, FR)
def generate_batch_EN_FR(data_batch):
    en_batch, fr_batch = [], []
    for (en_item, _, fr_item) in data_batch:        
        en_t = tensor_transform(en_item, mc.BOS_EN_IDX, mc.EOS_EN_IDX)
        fr_t = tensor_transform(fr_item, mc.BOS_FR_IDX, mc.EOS_FR_IDX)
        en_batch.append(en_t)
        fr_batch.append(fr_t)
    en_batch = pad_sequence(en_batch, padding_value=mc.PAD_EN_IDX, batch_first=False)
    fr_batch = pad_sequence(fr_batch, padding_value=mc.PAD_FR_IDX, batch_first=False)
    return en_batch, fr_batch

# Clase que implementa un custom dataset sobre la base de un iterador y no una lista iterable
class LanguageDataset(Dataset):
    
    def __init__(self, subset, length):

        super().__init__()
        self.length = length
        self.subset = subset

    def __getitem__(self, idx):
        return self.subset[idx]

    def __len__(self):
        return self.length

# Formato básico común para todos los gráficos de salida

plt.rc('font', size=8)
plt.rc('axes', titlesize=8)
plt.rc('axes', labelsize=8)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('legend', fontsize=8)
plt.rc('figure', titlesize=8)


# Gráfico que muestra las palabras más usadas en los diferentes vacabularios.

def plot_top_words(counter : Counter, vocab: Vocab, k=20, ax=None):
    top_k = counter.most_common(k)
    tokens, freqs = zip(*reversed(top_k))
    
    words = [vocab.lookup_token(token) for token in tokens]

    if ax is None:
        plt.barh(words, freqs, color='gray')
    else:
        ax.barh(words, freqs, color='gray')

# Gráfico que muestra el conteo de palabras y su frecuencia a lo largo de cada lenguaje.

def plot_corpus_counting_charts(en_counter, es_counter, fr_counter):
    fig = plt.figure(figsize=(10, 8))

    ax0 = fig.add_subplot(131)
    plot_top_words(en_counter, mc.en_vocabulary, ax=ax0)
    ax0.set_title("Top 20 English Words")
    ax0.set_xlabel("Raw Frequency")

    ax1 = fig.add_subplot(132)
    plot_top_words(es_counter, mc.es_vocabulary, ax=ax1)
    ax1.set_title("Top 20 Spanish Words")
    ax1.set_xlabel("Raw Frequency")

    ax2 = fig.add_subplot(133)
    plot_top_words(fr_counter, mc.fr_vocabulary, ax=ax2)
    ax2.set_title("Top 20 French Words")
    ax2.set_xlabel("Raw Frequency")

    plt.tight_layout()
    plt.savefig("tokens_frequency.svg")

# Grafica la longitud de las oraciones presentes en el corpus o su muestra.

def plot_corpus_lengths_charts(en_lengths, es_lengths, fr_lengths, max_seq_length):
    fig = plt.figure(figsize=(8, 10))
    ax0 = fig.add_subplot(311)
    ax0.hist(en_lengths, rwidth=0.8, color='gray')
    ax0.set_title("English Sentence Length")
    ax0.set_xlabel("# Tokens in Sentence")

    ax1 = fig.add_subplot(312)
    ax1.hist(es_lengths, rwidth=0.8, color='gray')
    ax1.set_title("Spanish Sentence Length")
    ax1.set_xlabel("# Tokens in Sentence")

    ax2 = fig.add_subplot(313)
    ax2.hist(fr_lengths, rwidth=0.8, color='gray')
    ax2.set_title("French Sentence Length")
    ax2.set_xlabel("# Tokens in Sentence")

    plt.tight_layout()
    plt.savefig("tokens_in_sentence.svg")

    plt.figure(figsize=(8,6))
    plt.hist2d(en_lengths, es_lengths, bins=max_seq_length-2, cmap='binary')
    plt.title("Joint Distribution of Sentence Lengths")
    plt.xlabel("# English Tokens")
    plt.ylabel("# Spanish Tokens")
    plt.colorbar()
    plt.savefig("joint_distribution_EN_ES.svg")

    plt.figure(figsize=(8,6))
    plt.hist2d(en_lengths, fr_lengths, bins=max_seq_length-2, cmap='binary')
    plt.title("Joint Distribution of Sentence Lengths")
    plt.xlabel("# English Tokens")
    plt.ylabel("# French Tokens")
    plt.colorbar()
    plt.savefig("joint_distribution_EN_FR.svg")

# define la cantidad de posiciones decimales que se usan en los gráficos
decimal_positions = 4

# Consolida los datos de las métricas de las diferentes iteraciones para los modelos de una sola tarea:
def history_to_table_single_task(name, language, source, epochs):
    result_MM = {}
    result_PF = {}
    for epoch in range(1, epochs + 1):
        result_MM[epoch] = dict(
                model_name = name,
                model_epoch = epoch,
                model_lang = language,
                trn_loss = round(source[epoch]['train_loss'], decimal_positions),
                trn_accm = round(source[epoch]['train_accm'], decimal_positions),
                trn_pplx = round(source[epoch]['train_pplx'], decimal_positions),
                val_loss = round(source[epoch]['valid_loss'], decimal_positions),
                val_accm = round(source[epoch]['valid_accm'], decimal_positions),
                val_pplx = round(source[epoch]['valid_pplx'], decimal_positions)

            )
        
        result_PF[epoch] = dict(
                model_name = name,
                model_epoch = epoch,
                model_lang = language,
                trn_wrkmem = round(source[epoch]['train_wrkmem'] / mc.size_to_MB, decimal_positions),
                trn_logmem = round(source[epoch]['train_logmem'] / mc.size_to_MB, decimal_positions),
                trn_elapsed_time  = round(source[epoch]['trn_elapsed_time'], decimal_positions),
                val_wrkmem = round(source[epoch]['valid_wrkmem'] / mc.size_to_MB, decimal_positions),
                val_logmem = round(source[epoch]['valid_logmem'] / mc.size_to_MB, decimal_positions),                
                val_elapsed_time  = round(source[epoch]['val_elapsed_time'], decimal_positions)
            )
    
    return result_MM, result_PF

# Consolida los datos de las métricas de las diferentes iteraciones para los modelos de doble tarea:

def history_to_table_dual_task(name, source, epochs):
    result_ES = {}
    result_FR = {}
    result_PF = {}

    for epoch in range(1, epochs + 1):
        result_ES[epoch] = dict(
                model_name = name,
                model_epoch = epoch,
                model_lang = 'ES',
                trn_loss = round(source[epoch]['train_loss_a'], decimal_positions),
                trn_accm = round(source[epoch]['train_acc_a'], decimal_positions),
                trn_pplx = round(np.exp(source[epoch]['train_loss_a']), decimal_positions),
                val_loss = round(source[epoch]['valid_loss_a'], decimal_positions),
                val_accm = round(source[epoch]['valid_acc_a'], decimal_positions),
                val_pplx  = round(np.exp(source[epoch]['valid_loss_a']), decimal_positions)
            )

        result_FR[epoch] = dict(
                model_name = name,
                model_epoch = epoch,
                model_lang = 'FR',
                trn_loss = round(source[epoch]['train_loss_b'], decimal_positions),
                trn_accm = round(source[epoch]['train_acc_b'], decimal_positions),
                trn_pplx = round(np.exp(source[epoch]['train_loss_b']), decimal_positions),
                val_loss = round(source[epoch]['valid_loss_b'], decimal_positions),
                val_accm = round(source[epoch]['valid_acc_b'], decimal_positions),
                val_pplx  = round(np.exp(source[epoch]['valid_loss_b']), decimal_positions)   
            )
        
        result_PF[epoch] = dict(
                model_name = name,
                model_epoch = epoch,
                model_lang = 'ES/FR',
                trn_wrkmem = round(source[epoch]['train_wrkmem'] / mc.size_to_MB, decimal_positions),
                trn_logmem = round(source[epoch]['train_logmem'] / mc.size_to_MB, decimal_positions),
                trn_elapsed_time  = round(source[epoch]['trn_elapsed_time'], decimal_positions),
                val_wrkmem = round(source[epoch]['valid_wrkmem'] / mc.size_to_MB, decimal_positions),
                val_logmem = round(source[epoch]['valid_logmem'] / mc.size_to_MB, decimal_positions),                    
                val_elapsed_time  = round(source[epoch]['val_elapsed_time'], decimal_positions)
            )
    
    return result_ES, result_FR, result_PF