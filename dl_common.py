# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: dl_common.py
# Descripción: Constantes y variables globales compartidas por todos los módulos
# Implementado por: Felipe Ramírez Herrera (basado en código de terceros)
# Curso Aprendizaje Profundo 1 y 2. Universidad de Valencia / ADEIT
# Ultima revisión: 11/04/2024 
# ----------------------------------------------------------------------------------------------------------------------------------

from enum import Enum
from typing import List
import pandas as pd

# Globales
NUMBER_OF_EPOCHS = 16
EARLY_STOPPING_EPOCHS = 5
CLIPPING_VALUE = 15 # Limite de crecimiento para los gradientes, usar None para desactivar
MAXIMUM_NUMBER_OF_SAMPLES = 500000
BATCH_SIZE = 32

GCCN_LR = 0.0001
TFMR_LR = 0.0001

# Nombres de archivos intermedios con preprocesamiento del corpus
pkl_dataset_file = "un_parallel_corpus.pkl"
en_vocab_file = 'un_parallel_vocab_en.pth'
es_vocab_file = 'un_parallel_vocab_es.pth'
fr_vocab_file = 'un_parallel_vocab_fr.pth'

# Nombres de archivos con conjuntos de datos para entrenamiento, evaluación e inferencia
full_data_file = 'un_parallel_full.pth'
trn_data_file = 'un_parallel_trn.pth'
val_data_file = 'un_parallel_val.pth'
tst_data_file = 'un_parallel_tst.pth'

un_ds = None
en_tokenizer = None
es_tokenizer = None
fr_tokenizer = None

Language = Enum('Language', ['EN', 'ES', 'FR'])

# Definición de tokens especiales.
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'  # Unknown token symbol
BOS_WORD = '<bos>'  # Begin-of-Sentence token symbol
EOS_WORD = '<eos>'  # End-of-Sentence token symbol

SPECIALS = [PAD_WORD, BOS_WORD, EOS_WORD,  UNK_WORD]

# Constante para convertir de bytes a megabytes
size_to_MB = 1024 * 1024

# Constantes para definir el rango de tamaños de oración preprocesados desde el corpus:
MINIMUM_ALLOWED_SIZE_OF_SEQ = 5
MAXIMUM_ALLOWED_SIZE_OF_SEQ = 100

# Tiempos de espera para descansar el GPU (ante eventuales sobrecalentamientos)
INNER_GPU_REST_TIME = 0 # Rest time between epochs
OUTER_GPU_REST_TIME = 0 # Rest time between model training / validation processes

# Tokens que representan constantes en los diferentes conjuntos de oraciones

PAD_EN_IDX = 0
BOS_EN_IDX = 0
EOS_EN_IDX = 0
UNK_EN_IDX = 0

PAD_ES_IDX = 0
BOS_ES_IDX = 0
EOS_ES_IDX = 0
UNK_ES_IDX = 0

PAD_FR_IDX = 0
BOS_FR_IDX = 0
EOS_FR_IDX = 0
UNK_FR_IDX = 0

# Variables que contienen los máximo y mínimo valores de la longitud de secuencia (hyperparam)
max_seq_length = 64 # Re-calculated further 
min_seq_length = 16 # Re-calculated further 

# Conjuntos de datos tokenizados
full_data = []
trn_subset = []
val_subset = []
tst_subset = []

# Tamaño de particiones de los datos
size_of_trn_set_size = 0
size_of_val_set_size = 0
size_of_tst_set_size = 0 

en_vocabulary = None
es_vocabulary = None
fr_vocabulary = None

# Tamaño de vocabularios (hyperparam)
en_vocab_size = 0
es_vocab_size = 0
fr_vocab_size = 0

batches_for_training = 0
batches_for_validation  = 0


# Patrones de checkpoits para modelos:

tfmr_st_es_model_path = 'tfmr_st_es_{0}.pt'
tfmr_st_fr_model_path = 'tfmr_st_fr_{0}.pt'
tfmr_dt_model_path = 'tfmr_dt_{0}.pt'
tfmr_dt_ws_model_path = 'tfmr_dt_ws_{0}.pt'

gcnn_st_es_model_path = 'gcnn_st_es_{0}.pt'
gcnn_st_fr_model_path = 'gcnn_st_fr_{0}.pt'
gcnn_dt_model_path = 'gcnn_dt_{0}.pt'
gccn_dt_ws_model_path = 'gcnn_dt_ws_{0}.pt'