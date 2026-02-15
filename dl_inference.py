# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: dl_inference.py
# Descripción: Programa que realiza la inferencia y calcula las métricas asociadas
# Implementado por: Felipe Ramírez Herrera (basado en código de terceros)
# Curso Aprendizaje Profundo 1 y 2. Universidad de Valencia / ADEIT
# Ultima revisión: 11/04/2024 
# ----------------------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import os
import dl_common as mc
import dl_gcnn_models as mg
import dl_tfmr_models as mt
import dl_xxxx_models as mm
import dl_greedy_search as mgs
import dl_common_processing as mp
import math
import pandas as pd
import torchtext
import torchtext.data
import torchtext.data.datasets_utils
import torchtext.datasets
from torch.utils.data import DataLoader
import random
import pickle
from collections import Counter 
from random import sample 
import warnings
warnings.filterwarnings('ignore')

max_number_of_samples = 10000 # Selecciona una cantidad especifíca de ejemplos

st_metrics = [] # Métricas para modelos de una tarea
dt_metrics = [] # Métricas para modelos de dos tareas

st_samples = [] # Ejemplos recolectados para PPT
dt_samples = [] # Ejemplos recolectados para PPT

specific_samples_set = [1, 3, 5, 6, 9, 10] # Use estos ejemplos para obtener los mismos resultados que en la PPT o [] para aproximar el BLEU general de la PPT

# Carga un modelo desde su último check point almacenado en disco
def load_model_from(model, filename):
    checkpoint = torch.load(filename.format(mc.NUMBER_OF_EPOCHS - 1))
    if (checkpoint['epoch'] == mc.NUMBER_OF_EPOCHS - 1):
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise RuntimeError("Checkpoint {0} is corrupt".format(mc.NUMBER_OF_EPOCHS -1))

# Carga los datos preprocesados y prepara el ambiente 

def execute_prepare_env():

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    torch.manual_seed(1234)

    if os.path.exists(mc.en_vocab_file):
        mc.en_vocabulary = torch.load(mc.en_vocab_file)
    else:
        raise Exception("")

    if os.path.exists(mc.es_vocab_file):
        mc.es_vocabulary = torch.load(mc.es_vocab_file)
    else:
        raise Exception("")

    if os.path.exists(mc.fr_vocab_file):
        mc.fr_vocabulary = torch.load(mc.fr_vocab_file)
    else:
        raise Exception("")


    print("EN Vocabulary Size = {0}".format(len(mc.en_vocabulary)))
    print("ES Vocabulary Size = {0}".format(len(mc.es_vocabulary)))
    print("FR Vocabulary Size = {0}".format(len(mc.fr_vocabulary)))


    mc.PAD_EN_IDX = mc.en_vocabulary[mc.PAD_WORD]
    mc.BOS_EN_IDX = mc.en_vocabulary[mc.BOS_WORD]
    mc.EOS_EN_IDX = mc.en_vocabulary[mc.EOS_WORD]
    mc.UNK_EN_IDX = mc.en_vocabulary[mc.UNK_WORD]


    mc.PAD_ES_IDX = mc.es_vocabulary[mc.PAD_WORD]
    mc.BOS_ES_IDX = mc.es_vocabulary[mc.BOS_WORD]
    mc.EOS_ES_IDX = mc.es_vocabulary[mc.EOS_WORD]
    mc.UNK_ES_IDX = mc.es_vocabulary[mc.UNK_WORD]

    mc.PAD_FR_IDX = mc.fr_vocabulary[mc.PAD_WORD]
    mc.BOS_FR_IDX = mc.fr_vocabulary[mc.BOS_WORD]
    mc.EOS_FR_IDX = mc.fr_vocabulary[mc.EOS_WORD]
    mc.UNK_FR_IDX = mc.fr_vocabulary[mc.UNK_WORD]

    mc.en_vocab_size = len(mc.en_vocabulary)
    mc.es_vocab_size = len(mc.es_vocabulary)
    mc.fr_vocab_size = len(mc.fr_vocabulary)


    max_len = 0
    min_len = 4500

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
            a = len(en_seq)
            b = len(es_seq)
            c = len(fr_seq) 
            max_len = max(max_len, a, b, c)    
            min_len = min(min_len, a, b, c)    
    else:
        raise Exception("")

    if (mc.max_seq_length < max_len):
        mc.max_seq_length = max_len + 2

    if (mc.min_seq_length > min_len):
        mc.min_seq_length = min_len + 2


    print("Selected Records: {0}".format(len(mc.full_data)))
    print("MAX SEQ LEN {0}".format(mc.max_seq_length))
    print("MIN SEQ LEN {0}".format(mc.min_seq_length))

    size_of_trn_set_size = len(mc.trn_subset)
    size_of_val_set_size = len(mc.val_subset)
    size_of_tst_set_size = len(mc.tst_subset)

    print("Training tuples: {0} Validation tuples: {1} Testing tuples: {2}".format(size_of_trn_set_size, size_of_val_set_size, size_of_tst_set_size))





def execute_tfrmr_inferences(tokenized_subset):

    # Hiperparámetros (los mismo del entrenamiento)
    EMB_SIZE = 256
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    # Estos son los modelos individuales
    # Recuerde que se contabilizan como una composición, es decir, se resuelve el problema usando dos modelos independientes.
    tfrmr_st_comp_es = mt.Seq2SeqTransformer("TFMR_EN_ES", NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, mc.en_vocab_size, mc.es_vocab_size, FFN_HID_DIM)
    tfrmr_st_comp_fr = mt.Seq2SeqTransformer("TFMR_EN_FR",NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, mc.en_vocab_size, mc.fr_vocab_size, FFN_HID_DIM)

    # Este es el modelo dual decoder
    tfrmr_dt = mt.DoubleTaskSeq2SeqTransformer("TFMR_DUAL_EN_ES_FR", NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, mc.en_vocab_size, mc.es_vocab_size, mc.fr_vocab_size, FFN_HID_DIM)
    # Este es el modelo dual decoder con weight-sharing
    tfrmr_dt_ws = mt.DoubleTaskSeq2SeqTransformer("TFMR_DUAL_EN_ES_FR (WS)", NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, mc.en_vocab_size, mc.es_vocab_size, mc.fr_vocab_size, FFN_HID_DIM, weight_sharing=True)

    tfrmr_st_comp_es.to(mm.DEVICE)
    tfrmr_st_comp_fr.to(mm.DEVICE)
    tfrmr_dt.to(mm.DEVICE)
    tfrmr_dt_ws.to(mm.DEVICE)


    # Carga el último checkpoint de cada modelo

    load_model_from(tfrmr_st_comp_es,  mc.tfmr_st_es_model_path)
    load_model_from(tfrmr_st_comp_fr,  mc.tfmr_st_fr_model_path)
    load_model_from(tfrmr_dt,  mc.tfmr_dt_model_path)
    load_model_from(tfrmr_dt_ws,  mc.tfmr_dt_ws_model_path)

    # Inferencia en GPU

    models = [tfrmr_st_comp_es, tfrmr_st_comp_fr]

    src_langs = [mc.Language.EN, mc.Language.EN]
    tgt_langs = [mc.Language.ES, mc.Language.FR]

    hist, smpl = mgs.run_tfrmr_st_inference(tokenized_subset, models, src_langs, tgt_langs, samples=specific_samples_set)

    st_metrics.append(pd.DataFrame.from_dict(hist, orient='index'))
    if len(smpl) > 0:
        st_samples.append(pd.DataFrame.from_dict(smpl, orient='index'))
            
    hist, smpl = mgs.run_tfrmr_dt_inference(tokenized_subset, [tfrmr_dt, tfrmr_dt_ws], samples=specific_samples_set)

    dt_metrics.append(pd.DataFrame.from_dict(hist, orient='index'))

    if len(smpl) > 0:
        dt_samples.append(pd.DataFrame.from_dict(smpl, orient='index'))

def execute_gcnn_inferences(tokenized_subset):

    # Hiperparámetros (los mismo del entrenamiento)
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
    gcnn_es_enc = mg.GatedConvEncoder(INPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_ENC_LAYERS, GCNN_ENC_KERNEL_SIZE, GCNN_ENC_DROPOUT, mm.DEVICE, mc.max_seq_length)
    gcnn_es_dec = mg.GatedConvDecoder(OUTPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_fr_enc = mg.GatedConvEncoder(INPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_ENC_LAYERS, GCNN_ENC_KERNEL_SIZE, GCNN_ENC_DROPOUT, mm.DEVICE, mc.max_seq_length)
    gccn_fr_dec = mg.GatedConvDecoder(OUTPUT_FR_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_st_es_comp_model = mg.GatedConvSeq2Seq("GCNN_ST_COMP_ES", gcnn_es_enc, gcnn_es_dec)
    gcnn_st_fr_comp_model = mg.GatedConvSeq2Seq("GCNN_ST_COMP_FR", gcnn_fr_enc, gccn_fr_dec)

    # Este es el modelo dual decoder
    gcnn_dt_en_enc = mg.GatedConvEncoder(INPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_ENC_LAYERS, GCNN_ENC_KERNEL_SIZE, GCNN_ENC_DROPOUT, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_es_dec = mg.GatedConvDecoder(OUTPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_fr_dec = mg.GatedConvDecoder(OUTPUT_FR_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_model = mg.GatedConvDualTaskSeq2Seq("GCNN_DT", gcnn_dt_en_enc, gcnn_dt_es_dec, gcnn_dt_fr_dec)

    # Este es el modelo dual decoder con weight-sharing
    gcnn_dt_ws_enc = mg.GatedConvEncoder(INPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_ENC_LAYERS, GCNN_ENC_KERNEL_SIZE, GCNN_ENC_DROPOUT, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_ws_es_dec = mg.GatedConvDecoder(OUTPUT_ES_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_ws_fr_dec = mg.GatedConvDecoder(OUTPUT_FR_VOCAB_DIM, GCNN_EMBDIM, GCNN_HID_DIM, GCNN_DEC_LAYERS, GCNN_DEC_KERNEL_SIZE, GCNN_DEC_DROPOUT, CNN_TRG_PAD_IDX, mm.DEVICE, mc.max_seq_length)
    gcnn_dt_ws_model = mg.GatedConvDualTaskSeq2Seq("GCNN_DT_WS", gcnn_dt_ws_enc, gcnn_dt_ws_es_dec, gcnn_dt_ws_fr_dec, shared_weights=True)

    # Carga el último checkpoint de cada modelo
    load_model_from(gcnn_st_es_comp_model,  mc.gcnn_st_es_model_path)
    load_model_from(gcnn_st_fr_comp_model,  mc.gcnn_st_fr_model_path)
    load_model_from(gcnn_dt_model,  mc.gcnn_dt_model_path)
    load_model_from(gcnn_dt_ws_model,  mc.gccn_dt_ws_model_path)

    gcnn_st_es_comp_model.to(mm.DEVICE)
    gcnn_st_fr_comp_model.to(mm.DEVICE)
    gcnn_dt_model.to(mm.DEVICE)
    gcnn_dt_ws_model.to(mm.DEVICE)

    # Inferencia en GPU

    models = [gcnn_st_es_comp_model, gcnn_st_fr_comp_model]

    src_langs = [mc.Language.EN, mc.Language.EN]
    tgt_langs = [mc.Language.ES, mc.Language.FR]

    hist, smpl = mgs.run_gccn_st_inference(tokenized_subset, models, src_langs, tgt_langs, samples=specific_samples_set)

    st_metrics.append(pd.DataFrame.from_dict(hist, orient='index'))
    
    if len(smpl) > 0:
        st_samples.append(pd.DataFrame.from_dict(smpl, orient='index'))

    hist, smpl = mgs.run_gccn_dt_inference(tokenized_subset, [gcnn_dt_model, gcnn_dt_ws_model], samples=specific_samples_set)

    dt_metrics.append(pd.DataFrame.from_dict(hist, orient='index'))

    if len(smpl) > 0:
        dt_samples.append(pd.DataFrame.from_dict(smpl, orient='index'))


execute_prepare_env()

tokenized_subset = []

random.seed(1234)

# Debido a capacidad de computo y tiempo insuficiente, se acota a max_number_of_samples ejemplos muestreados de forma aleatoria.

sampled_tst_subset = sample(mc.tst_subset, max_number_of_samples) 

for (src, tgt_ES, tgt_FR) in iter(sampled_tst_subset):

    src_EN_tokens = mc.en_vocabulary.lookup_tokens(src)
    tgt_ES_tokens = mc.es_vocabulary.lookup_tokens(tgt_ES)
    tgt_FR_tokens = mc.fr_vocabulary.lookup_tokens(tgt_FR)

    tokenized_subset.append((src, tgt_ES, tgt_FR, src_EN_tokens, tgt_ES_tokens,  tgt_FR_tokens))

# Se cargan los modelos y se ejecutan las mediciones sobre la inferencia:
execute_tfrmr_inferences(tokenized_subset)
execute_gcnn_inferences(tokenized_subset)

# Se recolectan y consolidan los resultados en archivos para construir la PPT

results_for_st = pd.concat(st_metrics)
results_for_st.to_csv("./st_inference.csv", sep=';', index=True, encoding='utf-8')

print(results_for_st)

results_for_dt = pd.concat(dt_metrics)
results_for_dt.to_csv("./dt_inference.csv", sep=';', index=True, encoding='utf-8')

print(results_for_dt)

if (len(st_samples) > 0):
    samples_for_st = pd.concat(st_samples)
    samples_for_st.to_csv("./st_samples.csv", sep=';', index=True, encoding='utf-8')
    print(samples_for_st)

if (len(dt_samples) > 0):
    samples_for_dt = pd.concat(dt_samples)
    samples_for_dt.to_csv("./dt_samples.csv", sep=';', index=True, encoding='utf-8')
    print(samples_for_dt)


        
