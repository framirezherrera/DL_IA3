# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: dl_main.py
# Descripción: Programa principal que entrena todos los modelos y calcula las métricas relacionadas
# Implementado por: Felipe Ramírez Herrera
# Curso Aprendizaje Profundo 1 y 2. Universidad de Valencia / ADEIT
# Ultima revisión: 11/04/2024 
# ----------------------------------------------------------------------------------------------------------------------------------


# !pip install numpy
# !pip install panda

# !pip install torch
# !pip install torchtext
# !pip install torchmetrics
# !pip install spacy

# !pip install jupyter
# !pip install ipywidgets

# !pip install editdistance
# !pip install six
# !pip install typeguard

# !pip install matplotlib
# !pip install seaborn
# !pip install scikit-learn

# !python -m spacy download en_core_web_sm
# !python -m spacy download es_core_news_sm
# !python -m spacy download fr_core_news_sm
# !pip install wget

# !pip install --upgrade --force-reinstall torchtext


import math
import random
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as multiproc

import torchtext
import torchtext.data
import torchtext.data.datasets_utils
import torchtext.datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torchtext.vocab import Vocab

import tqdm

import pathlib
import csv


import logging

import six
from typing import List, Tuple, Union
from argparse import Namespace
from tqdm.notebook import tqdm

import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import torchtext
import random
import pickle
from collections import Counter 
import itertools
import sys
from  torch.nn.utils.rnn import pad_sequence
import time
import datetime
warnings.filterwarnings('ignore')


import dl_common as mc
import dl_common_processing as mp
import dl_xxxx_models as mm
import dl_gcnn_models as mg
import dl_tfmr_models as mt
import dl_tfmr_routines as mtr
import dl_gcnn_routines as mgr


# -----------------------------------------------------------------------------------------------------------
# Particiones de entrenamiento
# -----------------------------------------------------------------------------------------------------------

# Modelos ST de EN a ES        
EN_to_ES_trnset = []
EN_to_ES_valset = []
EN_to_ES_tstset = []

# Modelos ST de EN a FR
EN_to_FR_trnset = []
EN_to_FR_valset = []
EN_to_FR_tstset = []

# Modelos DT de EN a ES y FR
EN_to_ES_and_FR_trnset = []
EN_to_ES_and_FR_valset = []
EN_to_ES_and_FR_tstset = []


# -----------------------------------------------------------------------------------------------------------
# Entrena y valida los modelos basados en Transformers
# -----------------------------------------------------------------------------------------------------------

def tfmr_exec():

    EMB_SIZE = 256
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3


    # Estos son los modelos individuales
    # Recuerde que se contabilizan como una composición, es decir, se resuelve el problema usando dos modelos independientes.
    tfrmr_st_comp_es = mt.Seq2SeqTransformer("TFMR_ST_COMP_ES", NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, mc.en_vocab_size, mc.es_vocab_size, FFN_HID_DIM)
    tfrmr_st_comp_fr = mt.Seq2SeqTransformer("TFMR_ST_COMP_FR",NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, mc.en_vocab_size, mc.fr_vocab_size, FFN_HID_DIM)
    # Modelo doble tarea
    tfrmr_dt = mt.DoubleTaskSeq2SeqTransformer("TFMR_DT", NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, mc.en_vocab_size, mc.es_vocab_size, mc.fr_vocab_size, FFN_HID_DIM)
    # Modelo doble tarea con weight-sharing
    tfrmr_dt_ws = mt.DoubleTaskSeq2SeqTransformer("TFMR_DT_WS", NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, mc.en_vocab_size, mc.es_vocab_size, mc.fr_vocab_size, FFN_HID_DIM, weight_sharing=True)

    random.seed(1234)
    torch.manual_seed(1234)
    tfrmr_st_comp_es.apply(mm.init_transformer_model)
    mm.count_params(tfrmr_st_comp_es)


    random.seed(1234)
    torch.manual_seed(1234)
    tfrmr_st_comp_fr.apply(mm.init_transformer_model)
    mm.count_params(tfrmr_st_comp_fr)


    random.seed(1234)
    torch.manual_seed(1234)
    tfrmr_dt.apply(mm.init_transformer_model)
    mm.count_params(tfrmr_dt)

    random.seed(1234)
    torch.manual_seed(1234)
    tfrmr_dt_ws.apply(mm.init_transformer_model)
    mm.count_params(tfrmr_dt_ws)

    tfrmr_st_comp_es_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=mc.PAD_ES_IDX)
    tfrmr_st_comp_fr_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=mc.PAD_ES_IDX)

    tfrmr_dt_es_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=mc.PAD_ES_IDX)
    tfrmr_dt_fr_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=mc.PAD_FR_IDX)

    tfrmr_dt_ws_es_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=mc.PAD_ES_IDX)
    tfrmr_dt_ws_fr_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=mc.PAD_FR_IDX)

    tfrmr_st_comp_es_opt = torch.optim.Adam(tfrmr_st_comp_es.parameters(), lr=mc.TFMR_LR, betas=(0.9, 0.98), eps=1e-9)
    tfrmr_st_comp_fr_opt = torch.optim.Adam(tfrmr_st_comp_fr.parameters(), lr=mc.TFMR_LR, betas=(0.9, 0.98), eps=1e-9)

    tfrmr_st_comp_es_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(tfrmr_st_comp_es_opt, 'min')
    tfrmr_st_comp_fr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(tfrmr_st_comp_fr_opt, 'min')

    tfrmr_dt_opt = torch.optim.Adam(tfrmr_dt.parameters(), lr=mc.TFMR_LR, betas=(0.9, 0.98), eps=1e-9)
    tfrmr_dt_ws_opt = torch.optim.Adam(tfrmr_dt_ws.parameters(), lr=mc.TFMR_LR, betas=(0.9, 0.98), eps=1e-9)

    tfrmr_dt_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(tfrmr_dt_opt, 'min')
    tfrmr_dt_ws_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(tfrmr_dt_ws_opt, 'min')
      
    tfmr_info_LP = []
    tfmr_info_PF = []

    histories = []

    random.seed(1234)
    torch.manual_seed(1234)
    tfrmr_st_comp_es = tfrmr_st_comp_es.to(mm.DEVICE)
    torch.compile(tfrmr_st_comp_es)
    history = mtr.run_single_transformer(tfrmr_st_comp_es, mc.tfmr_st_es_model_path, EN_to_ES_trnset, EN_to_ES_valset, tfrmr_st_comp_es_opt, tfrmr_st_comp_es_sch, tfrmr_st_comp_es_loss_fn, mc.PAD_EN_IDX, mc.PAD_ES_IDX)
    name = tfrmr_st_comp_es.getName()
    df = pd.DataFrame.from_dict(history, orient='index')   

    LP, PF = mp.history_to_table_single_task("TFMR_ST_COMP", "ES", history, mc.NUMBER_OF_EPOCHS)

    tfmr_info_LP.append(pd.DataFrame.from_dict(LP, orient='index'))
    tfmr_info_PF.append(pd.DataFrame.from_dict(PF, orient='index'))

    mm.plot_accuracy_curve(name, df)
    mm.plot_loss_curve(name, df)
    mm.plot_pplx_curve(name, df)
    histories.append(df)
    torch.cuda.empty_cache()
    mm.let_gpu_rest(mc.OUTER_GPU_REST_TIME)

    random.seed(1234)
    torch.manual_seed(1234)
    tfrmr_st_comp_fr = tfrmr_st_comp_fr.to(mm.DEVICE)
    torch.compile(tfrmr_st_comp_fr)
    history = mtr.run_single_transformer(tfrmr_st_comp_fr, mc.tfmr_st_fr_model_path,  EN_to_FR_trnset, EN_to_FR_valset, tfrmr_st_comp_fr_opt, tfrmr_st_comp_fr_sch, tfrmr_st_comp_fr_loss_fn, mc.PAD_EN_IDX, mc.PAD_FR_IDX)  
    name = tfrmr_st_comp_fr.getName()
    df = pd.DataFrame.from_dict(history, orient='index')   

    LP, PF = mp.history_to_table_single_task("TFMR_ST_COMP", "FR", history, mc.NUMBER_OF_EPOCHS)

    tfmr_info_LP.append(pd.DataFrame.from_dict(LP, orient='index'))
    tfmr_info_PF.append(pd.DataFrame.from_dict(PF, orient='index'))

    mm.plot_accuracy_curve(name, df)
    mm.plot_loss_curve(name, df)
    mm.plot_pplx_curve(name, df)
    histories.append(df)
    torch.cuda.empty_cache()
    mm.let_gpu_rest(mc.OUTER_GPU_REST_TIME)
    result = pd.concat(histories)
    
    result.to_csv("./metrics_tfmr_st_models.csv", sep=';', index=True, encoding='utf-8')
    print(result)

    tfrmr_dt = tfrmr_dt.to(mm.DEVICE)
    torch.compile(tfrmr_dt)

    tfrmr_dt_ws = tfrmr_dt_ws.to(mm.DEVICE)
    torch.compile(tfrmr_dt_ws)

    histories = []

    random.seed(1234)
    torch.manual_seed(1234)
    history = mtr.run_dual_transformer(tfrmr_dt, mc.tfmr_dt_model_path, EN_to_ES_and_FR_trnset, EN_to_ES_and_FR_valset,  tfrmr_dt_opt, tfrmr_dt_sch, tfrmr_dt_es_loss_fn, tfrmr_dt_fr_loss_fn, mc.PAD_EN_IDX, mc.PAD_ES_IDX, mc.PAD_FR_IDX)
    name = tfrmr_dt.getName()
    df = pd.DataFrame.from_dict(history, orient='index')


    LP_ES, LP_FR, PF = mp.history_to_table_dual_task("TFMR_DT", history, mc.NUMBER_OF_EPOCHS)

    tfmr_info_LP.append(pd.DataFrame.from_dict(LP_ES, orient='index'))
    tfmr_info_LP.append(pd.DataFrame.from_dict(LP_FR, orient='index'))
    tfmr_info_PF.append(pd.DataFrame.from_dict(PF, orient='index'))

    print(df)
    mm.plot_accuracy_curve_dual_transformer(name, df)
    mm.plot_accuracy_curve_dual_transformer_both(name, df)
    mm.plot_loss_curve_dual_transformer(name, df)
    mm.plot_loss_curve_dual_transformer_both(name, df)
    histories.append(df)


    random.seed(1234)
    torch.manual_seed(1234)
    history = mtr.run_dual_transformer(tfrmr_dt_ws, mc.tfmr_dt_ws_model_path, EN_to_ES_and_FR_trnset, EN_to_ES_and_FR_valset,  tfrmr_dt_ws_opt, tfrmr_dt_ws_sch, tfrmr_dt_ws_es_loss_fn, tfrmr_dt_ws_fr_loss_fn, mc.PAD_EN_IDX, mc.PAD_ES_IDX, mc.PAD_FR_IDX)
    name = tfrmr_dt_ws.getName()
    df = pd.DataFrame.from_dict(history, orient='index')

    LP_ES, LP_FR, PF = mp.history_to_table_dual_task("TFMR_DT_WS", history, mc.NUMBER_OF_EPOCHS)

    tfmr_info_LP.append(pd.DataFrame.from_dict(LP_ES, orient='index'))
    tfmr_info_LP.append(pd.DataFrame.from_dict(LP_FR, orient='index'))
    tfmr_info_PF.append(pd.DataFrame.from_dict(PF, orient='index'))

    print(df)
    mm.plot_accuracy_curve_dual_transformer(name, df)
    mm.plot_accuracy_curve_dual_transformer_both(name, df)
    mm.plot_loss_curve_dual_transformer(name, df)
    mm.plot_loss_curve_dual_transformer_both(name, df)
    histories.append(df)

    result = pd.concat(histories)
    result.to_csv("./metrics_tfmr_dt_models.csv", sep=';', index=True, encoding='utf-8')

    print(result)


    result_LP = pd.concat(tfmr_info_LP)
    result_LP.to_csv("./tfmr_LP.csv", sep=';', index=True, encoding='utf-8')

    print(result_LP)

    result_PF = pd.concat(tfmr_info_PF)
    result_PF.to_csv("./tfmr_PF.csv", sep=';', index=True, encoding='utf-8')

    print(result_PF)

# -----------------------------------------------------------------------------------------------------------
# Entrena y valida los modelos basados en Gated Convolutional Networks
# -----------------------------------------------------------------------------------------------------------


def gcnn_exec():

    INPUT_ES_VOCAB_DIM = mc.en_vocab_size
    OUTPUT_ES_VOCAB_DIM = mc.es_vocab_size
    OUTPUT_FR_VOCAB_DIM = mc.fr_vocab_size
    GCNN_EMBDIM = 256
    GCNN_HID_DIM = 512 # each conv. layer has 2 * hid_dim filters
    GCNN_ENC_LAYERS = 5 # number of conv. blocks in encoder
    GCNN_DEC_LAYERS = 5 # number of conv. blocks in decoder
    GCNN_ENC_KERNEL_SIZE = 3 # must be odd!
    GCNN_DEC_KERNEL_SIZE = 3 # can be even or odd
    GCNN_ENC_DROPOUT = 0.25
    GCNN_DEC_DROPOUT = 0.25

    CNN_TRG_PAD_IDX = mc.PAD_ES_IDX

    # Estos son los modelos individuales
    # Recuerde que se contabilizan como una composición, es decir, se resuelve el problema usando dos modelos independientes.
    gcnn_st_comp_es_enc = mg.GatedConvEncoder(INPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_ENC_LAYERS, GCNN_ENC_KERNEL_SIZE, GCNN_ENC_DROPOUT, mm.DEVICE, mc.max_seq_length)
    gcnn_st_comp_es_dec = mg.GatedConvDecoder(OUTPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_st_comp_es_model = mg.GatedConvSeq2Seq("GCNN_ST_COMP_ES", gcnn_st_comp_es_enc, gcnn_st_comp_es_dec)

    # Modelo single task para FR
    gcnn_st_comp_fr_enc = mg.GatedConvEncoder(INPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_ENC_LAYERS, GCNN_ENC_KERNEL_SIZE, GCNN_ENC_DROPOUT, mm.DEVICE, mc.max_seq_length)
    gcnn_st_comp_fr_dec = mg.GatedConvDecoder(OUTPUT_FR_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_st_comp_fr_model = mg.GatedConvSeq2Seq("GCNN_ST_COMP_FR", gcnn_st_comp_fr_enc, gcnn_st_comp_fr_dec)
    
    # Optimizadores:
    gcnn_st_comp_es_opt = optim.Adam(gcnn_st_comp_es_model.parameters(), lr=mc.GCCN_LR, betas=(0.9, 0.98), eps=1e-9)
    gcnn_st_comp_fr_opt = optim.Adam(gcnn_st_comp_fr_model.parameters(), lr=mc.GCCN_LR, betas=(0.9, 0.98), eps=1e-9)

    # Schedulers:
    gcnn_st_comp_es_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(gcnn_st_comp_es_opt, 'min')
    gcnn_st_comp_fr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(gcnn_st_comp_fr_opt, 'min')

    # Funciones de pérdida:
    gcnn_st_comp_es_loss_fn = nn.CrossEntropyLoss(ignore_index = CNN_TRG_PAD_IDX)
    gcnn_st_comp_fr_loss_fn = nn.CrossEntropyLoss(ignore_index = CNN_TRG_PAD_IDX)

    gcnn_dt_en_enc = mg.GatedConvEncoder(INPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_ENC_LAYERS, GCNN_ENC_KERNEL_SIZE, GCNN_ENC_DROPOUT, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_es_dec = mg.GatedConvDecoder(OUTPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_fr_dec = mg.GatedConvDecoder(OUTPUT_FR_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_model = mg.GatedConvDualTaskSeq2Seq("GCNN_DT", gcnn_dt_en_enc, gcnn_dt_es_dec, gcnn_dt_fr_dec)

    gcnn_dt_opt = optim.Adam(gcnn_dt_model.parameters(), lr=mc.GCCN_LR, betas=(0.9, 0.98), eps=1e-9)
    gcnn_dt_es_loss_fn = nn.CrossEntropyLoss(ignore_index = CNN_TRG_PAD_IDX)
    gcnn_dt_fr_loss_fn = nn.CrossEntropyLoss(ignore_index = CNN_TRG_PAD_IDX)
    gcnn_dt_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(gcnn_dt_opt, 'min')

    gcnn_dt_ws_enc = mg.GatedConvEncoder(INPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_ENC_LAYERS, GCNN_ENC_KERNEL_SIZE, GCNN_ENC_DROPOUT, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_ws_es_dec = mg.GatedConvDecoder(OUTPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_ws_fr_dec = mg.GatedConvDecoder(OUTPUT_FR_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_ws_model = mg.GatedConvDualTaskSeq2Seq("GCNN_DT_WS", gcnn_dt_ws_enc, gcnn_dt_ws_es_dec, gcnn_dt_ws_fr_dec, shared_weights=True)
    gcnn_dt_ws_opt = optim.Adam(gcnn_dt_ws_model.parameters(), lr=mc.GCCN_LR, betas=(0.9, 0.98), eps=1e-9)
    gcnn_dt_ws_es_loss_fn = nn.CrossEntropyLoss(ignore_index = CNN_TRG_PAD_IDX)
    gcnn_dt_ws_fr_loss_fn = nn.CrossEntropyLoss(ignore_index = CNN_TRG_PAD_IDX)
    gcnn_dt_ws_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(gcnn_dt_ws_opt, 'min')


    random.seed(1234)
    torch.manual_seed(1234)
    gcnn_st_comp_es_model.apply(mm.init_transformer_model)

    mm.count_params(gcnn_st_comp_es_model)

    random.seed(1234)
    torch.manual_seed(1234)
    gcnn_st_comp_fr_model.apply(mm.init_transformer_model)

    mm.count_params(gcnn_st_comp_fr_model)

    random.seed(1234)
    torch.manual_seed(1234)
    gcnn_dt_model.apply(mm.init_transformer_model)

    mm.count_params(gcnn_dt_model)


    random.seed(1234)
    torch.manual_seed(1234)
    gcnn_dt_ws_model.apply(mm.init_transformer_model)

    mm.count_params(gcnn_dt_ws_model)

    tfmr_info_LP = []
    tfmr_info_PF = []

    histories = []

    random.seed(1234)
    torch.manual_seed(1234)
    gcnn_st_comp_es_model = gcnn_st_comp_es_model.to(mm.DEVICE)
    torch.compile(gcnn_st_comp_es_model)
    history = mgr.run_single_cnn(gcnn_st_comp_es_model, mc.gcnn_st_es_model_path, EN_to_ES_trnset, EN_to_ES_valset, gcnn_st_comp_es_opt, gcnn_st_comp_es_sch, gcnn_st_comp_es_loss_fn, mc.PAD_EN_IDX, mc.PAD_ES_IDX)
       
    name = gcnn_st_comp_es_model.getName()
    df = pd.DataFrame.from_dict(history, orient='index')   

    LP, PF = mp.history_to_table_single_task("GCNN_ST_COMP", "ES", history, mc.NUMBER_OF_EPOCHS)

    tfmr_info_LP.append(pd.DataFrame.from_dict(LP, orient='index'))
    tfmr_info_PF.append(pd.DataFrame.from_dict(PF, orient='index'))

    mm.plot_accuracy_curve(name, df)
    mm.plot_loss_curve(name, df)
    mm.plot_pplx_curve(name, df)
    histories.append(df)
    torch.cuda.empty_cache()
    mm.let_gpu_rest(mc.OUTER_GPU_REST_TIME)

    random.seed(1234)
    torch.manual_seed(1234)
    gcnn_st_comp_fr_model = gcnn_st_comp_fr_model.to(mm.DEVICE)

    history = mgr.run_single_cnn(gcnn_st_comp_fr_model, mc.gcnn_st_fr_model_path, EN_to_FR_trnset, EN_to_FR_valset, gcnn_st_comp_fr_opt, gcnn_st_comp_fr_sch, gcnn_st_comp_fr_loss_fn, mc.PAD_EN_IDX, mc.PAD_FR_IDX)

    name = gcnn_st_comp_fr_model.getName()
    torch.compile(gcnn_st_comp_fr_model)
    df = pd.DataFrame.from_dict(history, orient='index')   

    LP, PF = mp.history_to_table_single_task("GCNN_ST_COMP", "FR", history, mc.NUMBER_OF_EPOCHS)

    tfmr_info_LP.append(pd.DataFrame.from_dict(LP, orient='index'))
    tfmr_info_PF.append(pd.DataFrame.from_dict(PF, orient='index'))

    mm.plot_accuracy_curve(name, df)
    mm.plot_loss_curve(name, df)
    mm.plot_pplx_curve(name, df)
    histories.append(df)
    torch.cuda.empty_cache()
    mm.let_gpu_rest(mc.OUTER_GPU_REST_TIME)

    result = pd.concat(histories)
    result.to_csv("./metrics_gcnn_st_models.csv", sep=';', index=True, encoding='utf-8')

    print(result)


    histories = []

    random.seed(1234)
    torch.manual_seed(1234)
    gcnn_dt_model = gcnn_dt_model.to(mm.DEVICE)
    torch.compile(gcnn_dt_model)
    history = mgr.run_dual_cnn(gcnn_dt_model, mc.gcnn_dt_model_path, EN_to_ES_and_FR_trnset, EN_to_ES_and_FR_valset,  gcnn_dt_opt, gcnn_dt_sch, gcnn_dt_es_loss_fn, gcnn_dt_fr_loss_fn, mc.PAD_EN_IDX, mc.PAD_ES_IDX, mc.PAD_FR_IDX)
    name = gcnn_dt_model.getName()
    df = pd.DataFrame.from_dict(history, orient='index')
    print(df)


    LP_ES, LP_FR, PF = mp.history_to_table_dual_task("GCNN_DT", history, mc.NUMBER_OF_EPOCHS)

    tfmr_info_LP.append(pd.DataFrame.from_dict(LP_ES, orient='index'))
    tfmr_info_LP.append(pd.DataFrame.from_dict(LP_FR, orient='index'))
    tfmr_info_PF.append(pd.DataFrame.from_dict(PF, orient='index'))


    mm.plot_accuracy_curve_dual_transformer(name, df)
    mm.plot_accuracy_curve_dual_transformer_both(name, df)
    mm.plot_loss_curve_dual_transformer(name, df)
    mm.plot_loss_curve_dual_transformer_both(name, df)
    histories.append(df)


    random.seed(1234)
    torch.manual_seed(1234)
    gcnn_dt_ws_model = gcnn_dt_ws_model.to(mm.DEVICE)
    torch.compile(gcnn_dt_ws_model)
    history = mgr.run_dual_cnn(gcnn_dt_ws_model, mc.gccn_dt_ws_model_path, EN_to_ES_and_FR_trnset, EN_to_ES_and_FR_valset,  gcnn_dt_ws_opt, gcnn_dt_ws_sch, gcnn_dt_ws_es_loss_fn, gcnn_dt_ws_fr_loss_fn, mc.PAD_EN_IDX, mc.PAD_ES_IDX, mc.PAD_FR_IDX)
    name = gcnn_dt_ws_model.getName()
    df = pd.DataFrame.from_dict(history, orient='index')
    print(df)

    LP_ES, LP_FR, PF = mp.history_to_table_dual_task("GCNN_DT_WS", history, mc.NUMBER_OF_EPOCHS)

    tfmr_info_LP.append(pd.DataFrame.from_dict(LP_ES, orient='index'))
    tfmr_info_LP.append(pd.DataFrame.from_dict(LP_FR, orient='index'))
    tfmr_info_PF.append(pd.DataFrame.from_dict(PF, orient='index'))

    mm.plot_accuracy_curve_dual_transformer(name, df)
    mm.plot_accuracy_curve_dual_transformer_both(name, df)
    mm.plot_loss_curve_dual_transformer(name, df)
    mm.plot_loss_curve_dual_transformer_both(name, df)
    histories.append(df)

    result = pd.concat(histories)
    result.to_csv("./metrics_gcnn_dt_models.csv", sep=';', index=True, encoding='utf-8')

    print(result)

    result_LP = pd.concat(tfmr_info_LP)
    result_LP.to_csv("./gcnn_LP.csv", sep=';', index=True, encoding='utf-8')

    print(result_LP)

    result_PF = pd.concat(tfmr_info_PF)
    result_PF.to_csv("./gcnn_PF.csv", sep=';', index=True, encoding='utf-8')

    print(result_PF)

# -----------------------------------------------------------------------------------------------------------
# Proceso principal
# -----------------------------------------------------------------------------------------------------------    

# Extrae los tokens de acuerdo al lenguaje de un conjunto (EN, ES, FR)
def yield_tokens(Lang: mc.Language=mc.Language.EN):
    for index, row in mc.un_ds.iterrows():     
         if (Lang == mc.Language.EN):
            yield mc.en_tokenizer(str(row["text_en"]))  
         if (Lang == mc.Language.ES):
            yield mc.es_tokenizer(str(row["text_es"]))
         if (Lang == mc.Language.FR):
           yield mc.fr_tokenizer(str(row["text_fr"]))     

if __name__ == '__main__':
    
    # Ojo que se da por un hecho que los archivos del corpus fueron descargados

    # El corpus debe ser el fully aligned de UN Parellel Corpus, se descarga de acá:
    # https://conferences.unite.un.org/UNCorpus/Home/DownloadOverview

    # Y se descomprime de la siguiente forma:
    # cat /home/framirez/translation_multilingual/UNv1.0.6way.tar.gz.* | tar -xzf -

    # No se puede automatizar la descarga del corpus por el tipo de enlace (onedrive)

    if os.path.exists(mc.pkl_dataset_file):
        mc.un_ds = pd.read_pickle(mc.pkl_dataset_file) 
    else:
        en_train_path = "./mnt/drive/UNv1.0.6way.en"
        es_train_path = "./mnt/drive/UNv1.0.6way.es"
        fr_train_path = "./mnt/drive/UNv1.0.6way.fr"
        df = mc.generate_examples(en_train_path, es_train_path, fr_train_path)
        df['text_en'] = df['text_en'].apply(lambda row: mp.clean_text(row))
        df['text_es'] = df['text_es'].apply(lambda row: mp.clean_text(row))
        df['text_fr'] = df['text_fr'].apply(lambda row: mp.clean_text(row))
        mc.un_ds = df
        mc.un_ds.to_pickle(mc.pkl_dataset_file)

    # Se acota el conjunto de datos a una fracción manejable en el hardware
    mc.un_ds = mc.un_ds.head(mc.MAXIMUM_NUMBER_OF_SAMPLES)

    # Se utilizan los tokenizadores de SpAcy

    mc.en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    mc.es_tokenizer = get_tokenizer('spacy', language='es_core_news_sm')
    mc.fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

    # Carga o genera el vocabulario en EN    
    if os.path.exists(mc.en_vocab_file):
        mc.en_vocabulary = torch.load(mc.en_vocab_file)
    else:
        mc.en_vocabulary = build_vocab_from_iterator(yield_tokens(mc.Language.EN), specials=mc.SPECIALS)
        torch.save(mc.en_vocabulary, mc.en_vocab_file)

    # Carga o genera el vocabulario en ES
    if os.path.exists(mc.es_vocab_file):
        mc.es_vocabulary = torch.load(mc.es_vocab_file)
    else:
        mc.es_vocabulary = build_vocab_from_iterator(yield_tokens(mc.Language.ES), specials=mc.SPECIALS)
        torch.save(mc.es_vocabulary, mc.es_vocab_file)

    # Carga o genera el vocabulario en FR
    if os.path.exists(mc.fr_vocab_file):
        mc.fr_vocabulary = torch.load(mc.fr_vocab_file)
    else:
        mc.fr_vocabulary = build_vocab_from_iterator(yield_tokens(mc.Language.FR), specials=mc.SPECIALS)
        torch.save(mc.fr_vocabulary, mc.fr_vocab_file)

    # Asigna símbolos especiales con respecto al vocabulario EN
    mc.PAD_EN_IDX = mc.en_vocabulary[mc.PAD_WORD]
    mc.BOS_EN_IDX = mc.en_vocabulary[mc.BOS_WORD]
    mc.EOS_EN_IDX = mc.en_vocabulary[mc.EOS_WORD]
    mc.UNK_EN_IDX = mc.en_vocabulary[mc.UNK_WORD]

    # Asigna símbolos especiales con respecto al vocabulario ES
    mc.PAD_ES_IDX = mc.es_vocabulary[mc.PAD_WORD]
    mc.BOS_ES_IDX = mc.es_vocabulary[mc.BOS_WORD]
    mc.EOS_ES_IDX = mc.es_vocabulary[mc.EOS_WORD]
    mc.UNK_ES_IDX = mc.es_vocabulary[mc.UNK_WORD]

    # Asigna símbolos especiales con respecto al vocabulario FR
    mc.PAD_FR_IDX = mc.fr_vocabulary[mc.PAD_WORD]
    mc.BOS_FR_IDX = mc.fr_vocabulary[mc.BOS_WORD]
    mc.EOS_FR_IDX = mc.fr_vocabulary[mc.EOS_WORD]
    mc.UNK_FR_IDX = mc.fr_vocabulary[mc.UNK_WORD]

    # Se determinan los tamaños de los vocabularios (hiperparametros)
    mc.en_vocab_size = len(mc.en_vocabulary) 
    mc.es_vocab_size = len(mc.es_vocabulary) 
    mc.fr_vocab_size = len(mc.fr_vocabulary)

    # Se despliegan los valores de los tamaños de los vocabularios
    print("EN Vocabulary Size = {0}".format(mc.en_vocab_size))
    print("ES Vocabulary Size = {0}".format(mc.es_vocab_size))
    print("FR Vocabulary Size = {0}".format(mc.fr_vocab_size))

    # Se imprime los caracteres especiales
    print("EN: PAD = {0} BOS = {1} EOS = {2} UNK = {3}".format(mc.PAD_EN_IDX, mc.BOS_EN_IDX, mc.EOS_EN_IDX, mc.UNK_EN_IDX))
    print("ES: PAD = {0} BOS = {1} EOS = {2} UNK = {3}".format(mc.PAD_ES_IDX, mc.BOS_ES_IDX, mc.EOS_ES_IDX, mc.UNK_ES_IDX))
    print("FR: PAD = {0} BOS = {1} EOS = {2} UNK = {3}".format(mc.PAD_FR_IDX, mc.BOS_FR_IDX, mc.EOS_FR_IDX, mc.UNK_FR_IDX))


    random.seed(1234)
    torch.manual_seed(1234)

    # Se determinan las longitudes de las sentencias

    max_len = 0
    min_len = 4500

    en_counter = Counter()
    es_counter = Counter()
    fr_counter = Counter()

    en_lengths = []
    es_lengths = []
    fr_lengths = []

    # Se cargan o crean los archivos de particiones del conjunto de datos
    fe = os.path.exists(mc.full_data_file)
    te = os.path.exists(mc.trn_data_file)
    ve = os.path.exists(mc.val_data_file)
    tt = os.path.exists(mc.tst_data_file)

    if fe and te and ve and tt:
        with open(mc.full_data_file, 'rb') as fp:
                mc.full_data = pickle.load(fp)
        with open(mc.trn_data_file, 'rb') as fp:
                mc.trn_subset = pickle.load(fp)
        with open(mc.val_data_file, 'rb') as fp:
                mc.val_subset = pickle.load(fp)        
        with open(mc.tst_data_file, 'rb') as fp:
                mc.tst_subset = pickle.load(fp)           
        for (en_seq,es_seq,fr_seq) in mc.full_data:     
            en_counter.update(en_seq)
            es_counter.update(es_seq)
            fr_counter.update(fr_seq)
            a = len(en_seq)
            b = len(es_seq)
            c = len(fr_seq) 
            en_lengths.append(a)
            es_lengths.append(b)
            fr_lengths.append(c)
            max_len = max(max_len, a, b, c)    
            min_len = min(min_len, a, b, c)    
    else:
        for idx, row in mc.un_ds.iterrows():
            
            en_exp = row["text_en"].rstrip("\n")
            es_exp = row["text_es"].rstrip("\n")
            fr_exp = row["text_fr"].rstrip("\n")

            en_seq = [mc.en_vocabulary[token] for token in mc.en_tokenizer(en_exp)]
            es_seq = [mc.es_vocabulary[token] for token in mc.es_tokenizer(es_exp)]
            fr_seq = [mc.fr_vocabulary[token] for token in mc.fr_tokenizer(fr_exp)]

            a = len(en_seq)
            b = len(es_seq)
            c = len(fr_seq) 

            seq_min = min(a, b, c)
            seq_max = max(a, b, c)

            if seq_min  >= mc.MINIMUM_ALLOWED_SIZE_OF_SEQ and seq_max <= mc.MAXIMUM_ALLOWED_SIZE_OF_SEQ: 
                
                en_counter.update(en_seq)
                es_counter.update(es_seq)
                fr_counter.update(fr_seq)
        
                en_lengths.append(a)
                es_lengths.append(b)
                fr_lengths.append(c)

                max_len = max(max_len, seq_max)
                min_len = min(min_len, seq_min) 
                mc.full_data.append((en_seq, es_seq, fr_seq))

        # Las particiones son 10% para Test, 70% Training y 20% para Validación
        mc.trn_subset, mc.val_subset = train_test_split(mc.full_data, test_size=0.3, train_size=0.7, random_state=1234, shuffle=True)
        mc.val_subset, mc.tst_subset = train_test_split(mc.val_subset, test_size=0.33, train_size=0.67, random_state=1234, shuffle=True)

        with open(mc.full_data_file, 'wb') as fp:
            pickle.dump(mc.full_data, fp)
        with open(mc.trn_data_file, 'wb') as fp:
            pickle.dump(mc.trn_subset, fp)
        with open(mc.val_data_file, 'wb') as fp:
            pickle.dump(mc.val_subset, fp)
        with open(mc.tst_data_file, 'wb') as fp:
            pickle.dump(mc.tst_subset, fp)

    if (mc.max_seq_length < max_len):
        mc.max_seq_length = max_len + 2

    if (mc.min_seq_length > min_len):
        mc.min_seq_length = min_len + 2


    print("Selected Records: {0}".format(len(mc.full_data)))
    print("MAX SEQ LEN {0}".format(mc.max_seq_length))
    print("MIN SEQ LEN {0}".format(mc.min_seq_length))

    mc.size_of_trn_set_size = len(mc.trn_subset)
    mc.size_of_val_set_size = len(mc.val_subset)
    mc.size_of_tst_set_size = len(mc.tst_subset)

    print("Training tuples: {0} Validation tuples: {1} Testing tuples: {2}".format(mc.size_of_trn_set_size, mc.size_of_val_set_size, mc.size_of_tst_set_size))

    # Se preparan los conjuntos de entrenamiento, validación y prueba para los diferentes ciclos de aprendizaje de los modelos ST (ES)
    EN_to_ES_trnset = DataLoader(mp.LanguageDataset(mc.trn_subset, mc.size_of_trn_set_size), batch_size=mc.BATCH_SIZE, shuffle=True, collate_fn=mp.generate_batch_EN_ES)
    EN_to_ES_valset = DataLoader(mp.LanguageDataset(mc.val_subset, mc.size_of_val_set_size), batch_size=mc.BATCH_SIZE, shuffle=False, collate_fn=mp.generate_batch_EN_ES)
    EN_to_ES_tstset = DataLoader(mp.LanguageDataset(mc.tst_subset, mc.size_of_tst_set_size), batch_size=mc.BATCH_SIZE, shuffle=False, collate_fn=mp.generate_batch_EN_ES)

    # Se preparan los conjuntos de entrenamiento, validación y prueba para los diferentes ciclos de aprendizaje de los modelos ST (FR)
    EN_to_FR_trnset = DataLoader(mp.LanguageDataset(mc.trn_subset, mc.size_of_trn_set_size), batch_size=mc.BATCH_SIZE, shuffle=True, collate_fn=mp.generate_batch_EN_FR)
    EN_to_FR_valset = DataLoader(mp.LanguageDataset(mc.val_subset, mc.size_of_val_set_size), batch_size=mc.BATCH_SIZE, shuffle=False, collate_fn=mp.generate_batch_EN_FR)
    EN_to_FR_tstset = DataLoader(mp.LanguageDataset(mc.tst_subset, mc.size_of_tst_set_size), batch_size=mc.BATCH_SIZE, shuffle=False, collate_fn=mp.generate_batch_EN_FR)

    # Se preparan los conjuntos de entrenamiento, validación y prueba para los diferentes ciclos de aprendizaje de los modelos DT (ES y FR)
    EN_to_ES_and_FR_trnset = DataLoader(mp.LanguageDataset(mc.trn_subset, mc.size_of_trn_set_size), batch_size=mc.BATCH_SIZE, shuffle=True, collate_fn=mp.generate_batch)
    EN_to_ES_and_FR_valset = DataLoader(mp.LanguageDataset(mc.val_subset, mc.size_of_val_set_size), batch_size=mc.BATCH_SIZE, shuffle=False, collate_fn=mp.generate_batch)
    EN_to_ES_and_FR_tstset = DataLoader(mp.LanguageDataset(mc.tst_subset, mc.size_of_tst_set_size), batch_size=mc.BATCH_SIZE, shuffle=False, collate_fn=mp.generate_batch)

    print("size_of_trn_set_size", mc.size_of_trn_set_size)
    print("size_of_val_set_size", mc.size_of_val_set_size)
    print("size_of_tst_set_size", mc.size_of_tst_set_size)

    mc.batches_for_training = math.ceil(mc.size_of_trn_set_size / mc.BATCH_SIZE)
    mc.batches_for_validation = math.ceil(mc.size_of_val_set_size / mc.BATCH_SIZE)

    # Se indican las cantidades de batches por conjunto.
    print("batches_for_training", mc.batches_for_training)
    print("batches_for_validation", mc.batches_for_validation)

    # Se generan algunas graficas relacionadas con el dataset utilizado
    mp.plot_corpus_counting_charts(en_counter, es_counter, fr_counter)
    mp.plot_corpus_lengths_charts(en_lengths, es_lengths, fr_lengths, mc.max_seq_length)

    # Se ejecutan los ciclos de entrenamiento de los modelos basados en transformers
    tfmr_exec()
    # Se ejecutan los ciclos de entrenamiento de los modelos basados en gated convolutional networks
    gcnn_exec()