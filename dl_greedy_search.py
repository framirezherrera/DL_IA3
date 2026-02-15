# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: dl_gcnn_routines.py
# Descripción: Búsqueda ávida para implementar la inferencia (traducción)
# Implementado por: Felipe Ramírez Herrera 
# Curso Aprendizaje Profundo 1 y 2. Universidad de Valencia / ADEIT
# Ultima revisión: 11/04/2024 
# ----------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu

import dl_common as mc
import dl_xxxx_models as mm
import dl_gcnn_models as mg
import dl_tfmr_models as mt
import dl_tfmr_routines as mtr
import dl_common_processing as mp
import datetime
from timeit import default_timer as timer


# Esta rutina permite eliminar todos aquellos símbolos que son utilizados para entrenar 
# y construir el dataset, pero que no son relevantes para presentarlos como parte del 
# resultado o como entrada de la métrica BLUE-4

def clean_seq_of_tokens(seq):
    if ' ' in seq:
        seq.remove(' ')
    if mc.BOS_WORD in seq:
        seq.remove(mc.BOS_WORD)
    if mc.EOS_WORD in seq:
        seq.remove(mc.EOS_WORD)
    if mc.PAD_WORD in seq:    
        seq.remove(mc.PAD_WORD)
    return seq



#    Decodifica secuencias utilizando el algoritmo greedy para un modelo de Transformer.

#    Parámetros:
#        model (mt.Seq2SeqTransformer): Modelo de Transformer Seq2Seq a utilizar para la decodificación.
#        src (torch.Tensor): Tensor de entrada.
#        src_mask (torch.BoolTensor): Máscara para la entrada.
#        max_len (int): Longitud máxima de la secuencia de salida.
#        bos_symbol (int): Símbolo de inicio de la secuencia.
#        pad_symbol (int): Símbolo de relleno.
#        eos_symbol (int): Símbolo de fin de la secuencia.

#    Retorna:
#        torch.Tensor: Tensor que representa la secuencia decodificada.

def tfrmr_st_greedy_decode(model : mt.Seq2SeqTransformer, src, src_mask, max_len, bos_symbol, pad_symbol, eos_symbol):
    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1, device=mm.DEVICE).fill_(bos_symbol).type(torch.long)
    for i in range(max_len-1):
        tgt_mask = mtr.generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        ys = torch.cat([ys, torch.ones(1, 1, device=mm.DEVICE).type_as(src.data).fill_(next_word)], dim=0)

        if next_word == eos_symbol:
            break

    return ys



#    Decodifica secuencias utilizando el algoritmo greedy para un modelo de Transformer de doble tarea.
#    Parámetros:
#        model (mt.DoubleTaskSeq2SeqTransformer): Modelo de Transformer de doble tarea a utilizar para la decodificación.
#        src (torch.Tensor): Tensor de entrada.
#        src_mask (torch.BoolTensor): Máscara para la entrada.
#        max_len (int): Longitud máxima de la secuencia de salida.
#        bos_symbol (int): Símbolo de inicio de la secuencia.
#        pad_symbol (int): Símbolo de relleno.
#        eos_symbol (int): Símbolo de fin de la secuencia.
#    Retorna:
#        torch.Tensor, torch.Tensor: Tensores que representan las secuencias decodificadas para las dos tareas.


def tfrmr_dt_greedy_decode(model : mt.DoubleTaskSeq2SeqTransformer, src, src_mask, max_len, bos_symbol, pad_symbol, eos_symbol):   
    src_padding_mask = mtr.get_padding_mask(src, mc.PAD_ES_IDX)

    memory = model.encode(src, src_mask)

    y_seq_lang_a = torch.ones(1, 1, device=mm.DEVICE).fill_(bos_symbol).type(torch.long)
    y_seq_lang_b = torch.ones(1, 1, device=mm.DEVICE).fill_(bos_symbol).type(torch.long)

    no_more_for_a = False
    no_more_for_b = False

    for i in range(max_len - 1):
        if (not no_more_for_a):

            tgt_a_mask = mtr.generate_square_subsequent_mask(y_seq_lang_a.size(0)).type(torch.bool)
            tgt_a_padding_mask = mtr.get_padding_mask(y_seq_lang_a, mc.PAD_ES_IDX)

            out_lang_a = model.decode_a(y_seq_lang_a, memory, tgt_a_mask, tgt_a_padding_mask, src_padding_mask)
            out_lang_a = out_lang_a.transpose(0, 1)

            prob_for_lang_a = model.generator_a(out_lang_a[:, -1])

            _, next_word = torch.max(prob_for_lang_a, dim=1)

            next_word = next_word.item()

            y_seq_lang_a = torch.cat([y_seq_lang_a, torch.ones(1, 1, device=mm.DEVICE).type_as(src.data).fill_(next_word)], dim=0)

            if next_word == eos_symbol:
                no_more_for_a = True

        if (not no_more_for_b):

            tgt_b_mask = mtr.generate_square_subsequent_mask(y_seq_lang_b.size(0)).type(torch.bool)
            tgt_b_padding_mask = mtr.get_padding_mask(y_seq_lang_b, mc.PAD_FR_IDX)

            out_lang_b = model.decode_b(y_seq_lang_b, memory, tgt_b_mask, tgt_b_padding_mask, src_padding_mask)
            out_lang_b = out_lang_b.transpose(0, 1)

            prob_for_lang_b = model.generator_b(out_lang_b[:, -1])

            _, next_word = torch.max(prob_for_lang_b, dim=1)

            next_word = next_word.item()

            y_seq_lang_b = torch.cat([y_seq_lang_b, torch.ones(1, 1, device=mm.DEVICE).type_as(src.data).fill_(next_word)], dim=0)

            if next_word == eos_symbol:
                no_more_for_b = True

        if no_more_for_a and no_more_for_b:
            break
               
    return y_seq_lang_a, y_seq_lang_b

#    Decodifica secuencias utilizando el algoritmo greedy para un modelo de GCNN de una tarea.
#    Parámetros:
#        model (mt.GatedConvSeq2Seq): Modelo de GCNN de una tarea a utilizar para la decodificación.
#        src (torch.Tensor): Tensor de entrada.
#        max_len (int): Longitud máxima de la secuencia de salida.
#        bos_symbol (int): Símbolo de inicio de la secuencia.
#        pad_symbol (int): Símbolo de relleno.
#        eos_symbol (int): Símbolo de fin de la secuencia.
#    Retorna:
#        torch.Tensor, torch.Tensor: Tensores que representan la secuencia decodificada.


def gccn_st_greedy_decode(model : mg.GatedConvSeq2Seq, src, max_len, bos_symbol, pad_symbol, eos_symbol):

    latent, combined = model.encode(src)

    trg_indexes = [bos_symbol]

    for i in range(max_len-1):
      
        trg_tensor = torch.tensor(trg_indexes, dtype=torch.long, device=mm.DEVICE).unsqueeze(0)

        output, _ = model.decode(trg_tensor, latent, combined)
      
        pred_token = output.argmax(2)[:, -1]
        pred_token = pred_token.item()
        
        trg_indexes.append(pred_token)

        if pred_token == eos_symbol:
            break
   

    return trg_indexes

#    Decodifica secuencias utilizando el algoritmo greedy para un modelo de GCNN de doble tarea.
#    Parámetros:
#        model (mt.GatedConvDualTaskSeq2Seq): Modelo de GCNN de doble tarea a utilizar para la decodificación.
#        src (torch.Tensor): Tensor de entrada.
#        max_len (int): Longitud máxima de la secuencia de salida.
#        bos_symbol (int): Símbolo de inicio de la secuencia.
#        pad_symbol (int): Símbolo de relleno.
#        eos_symbol (int): Símbolo de fin de la secuencia.
#    Retorna:
#        torch.Tensor, torch.Tensor: Tensores que representan las secuencias decodificadas para las dos tareas.

def gccn_dt_greedy_decode(model : mg.GatedConvDualTaskSeq2Seq, src, max_len, bos_symbol, pad_symbol, eos_symbol):

    latent, combined = model.encode(src)

    tgt_a_result = [bos_symbol]
    tgt_b_result = [bos_symbol]

    no_more_for_a = False
    no_more_for_b = False

    for i in range(max_len-1):
      
        if (not no_more_for_a):
            tgt_a = torch.tensor(tgt_a_result, dtype=torch.long, device=mm.DEVICE).unsqueeze(0)

            output_lang_a, _ = model.decode_for_task_a(tgt_a, latent, combined)
     
            pred_token = output_lang_a.argmax(2)[:, -1].item()

            tgt_a_result.append(pred_token)

            if pred_token == eos_symbol:
                no_more_for_a = True

        if (not no_more_for_b):
            tgt_b = torch.tensor(tgt_b_result, dtype=torch.long, device=mm.DEVICE).unsqueeze(0)

            output_lang_b, _ = model.decode_for_task_b(tgt_b, latent, combined)
     
            pred_token = output_lang_b.argmax(2)[:, -1].item()

            tgt_b_result.append(pred_token)

            if pred_token == eos_symbol:
                no_more_for_b = True

        if (no_more_for_a and no_more_for_b):
            break

    return tgt_a_result, tgt_b_result



#    Traduce una secuencia utilizando un modelo Seq2Seq Transformer.

#    Parámetros:
#        model (mt.Seq2SeqTransformer): Modelo Seq2Seq Transformer a utilizar para la traducción.
#        sequence (list): Secuencia de entrada a traducir.
#        tgt_lang (mc.Language): Lenguaje objetivo de la traducción.
#        translation_gap (int, opcional): Brecha de traducción para permitir espacio adicional en la secuencia de salida.

#    Retorna:
#        torch.Tensor: Secuencia traducida.
 
def tfrmr_st_translate_seq(model: mt.Seq2SeqTransformer, sequence, tgt_lang, translation_gap = 3):
    model.eval()
   
    bos_symbol = 0
    pad_symbol = 0
    eos_symbol = 0

    match (tgt_lang):
        case mc.Language.EN:
            tgt_vocab = mc.en_vocabulary
            bos_symbol = mc.BOS_EN_IDX
            pad_symbol = mc.PAD_EN_IDX
            eos_symbol = mc.EOS_EN_IDX
        case mc.Language.ES:
            tgt_vocab = mc.es_vocabulary
            bos_symbol = mc.BOS_ES_IDX
            pad_symbol = mc.PAD_ES_IDX
            eos_symbol = mc.EOS_ES_IDX
        case mc.Language.FR:
            tgt_vocab = mc.fr_vocabulary
            bos_symbol = mc.BOS_FR_IDX
            pad_symbol = mc.PAD_FR_IDX
            eos_symbol = mc.EOS_FR_IDX
        case _:
            raise RuntimeError("invalid tgt language")

    with torch.no_grad():

        src_seq = torch.tensor(sequence, dtype=torch.long, device=mm.DEVICE)
        src_seq = src_seq.view(-1, 1)
        num_tokens = src_seq.shape[0]
        src_seq_mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool, device=mm.DEVICE)
        tgt_tokens = tfrmr_st_greedy_decode(model,  src_seq, src_seq_mask, max_len=num_tokens + translation_gap, bos_symbol=bos_symbol, pad_symbol=pad_symbol, eos_symbol=eos_symbol).flatten()

    return tgt_tokens


#    Traduce una secuencia utilizando un modelo DoubleTaskSeq2SeqTransformer Transformer.

#    Parámetros:
#        model (mt.DoubleTaskSeq2SeqTransformer): Modelo DoubleTaskSeq2SeqTransformer Transformer a utilizar para la traducción.
#        sequence (list): Secuencia de entrada a traducir.
#        translation_gap (int, opcional): Brecha de traducción para permitir espacio adicional en la secuencia de salida.

#    Retorna:
#        torch.Tensor: Secuencia traducida.

def tfrmr_dt_translate_seq(model: mt.DoubleTaskSeq2SeqTransformer, sequence, translation_gap = 3):
    model.eval()

    bos_symbol = mc.BOS_EN_IDX
    pad_symbol = mc.PAD_EN_IDX
    eos_symbol = mc.EOS_EN_IDX

    with torch.no_grad():

        src_seq = torch.tensor(sequence, dtype=torch.long, device=mm.DEVICE)
        src_seq = src_seq.view(-1, 1)
        num_tokens = src_seq.shape[0]

        src_seq_mask = torch.zeros(num_tokens, num_tokens, dtype=torch.bool, device=mm.DEVICE)
        
        tgt_a, tgt_b = tfrmr_dt_greedy_decode(model,  src_seq, src_seq_mask, max_len=num_tokens + translation_gap, bos_symbol=bos_symbol, pad_symbol=pad_symbol, eos_symbol=eos_symbol)
        tgt_a, tgt_b = tgt_a.flatten(), tgt_b.flatten()

    return tgt_a, tgt_b

#    Traduce una secuencia utilizando un modelo GatedConvSeq2Seq.

#    Parámetros:
#        model (mt.GatedConvSeq2Seq): Modelo GatedConvSeq2Seq a utilizar para la traducción.
#        sequence (list): Secuencia de entrada a traducir.
#        tgt_lang (mc.Language): Lenguaje objetivo de la traducción.
#        translation_gap (int, opcional): Brecha de traducción para permitir espacio adicional en la secuencia de salida.

#    Retorna:
#        torch.Tensor: Secuencia traducida.

def gcnn_st_translate_seq(model: mg.GatedConvSeq2Seq, sequence, tgt_lang, translation_gap = 3):
    model.eval()
   
    bos_symbol = 0
    pad_symbol = 0
    eos_symbol = 0

    match (tgt_lang):
        case mc.Language.EN:
            tgt_vocab = mc.en_vocabulary
            bos_symbol = mc.BOS_EN_IDX
            pad_symbol = mc.PAD_EN_IDX
            eos_symbol = mc.EOS_EN_IDX
        case mc.Language.ES:
            tgt_vocab = mc.es_vocabulary
            bos_symbol = mc.BOS_ES_IDX
            pad_symbol = mc.PAD_ES_IDX
            eos_symbol = mc.EOS_ES_IDX
        case mc.Language.FR:
            tgt_vocab = mc.fr_vocabulary
            bos_symbol = mc.BOS_FR_IDX
            pad_symbol = mc.PAD_FR_IDX
            eos_symbol = mc.EOS_FR_IDX
        case _:
            raise RuntimeError("invalid tgt language")

    with torch.no_grad():
        src_seq = torch.tensor(sequence, dtype=torch.long, device=mm.DEVICE).unsqueeze(0)
        num_tokens = len(sequence)
        tgt_tokens = gccn_st_greedy_decode(model,  src_seq, max_len=num_tokens + translation_gap, bos_symbol=bos_symbol, pad_symbol=pad_symbol, eos_symbol=eos_symbol)
    return tgt_tokens


#    Traduce una secuencia utilizando un modelo GatedConvDualTaskSeq2Seq.

#    Parámetros:
#        model (mt.GatedConvDualTaskSeq2Seq): Modelo GatedConvDualTaskSeq2Seq a utilizar para la traducción.
#        sequence (list): Secuencia de entrada a traducir.
#        translation_gap (int, opcional): Brecha de traducción para permitir espacio adicional en la secuencia de salida.

#    Retorna:
#        torch.Tensor: Secuencia traducida.

def gcnn_dt_translate_seq(model: mg.GatedConvDualTaskSeq2Seq, sequence, translation_gap = 3):
    model.eval()
   
    bos_symbol = mc.BOS_EN_IDX
    pad_symbol = mc.PAD_EN_IDX
    eos_symbol = mc.EOS_EN_IDX

    with torch.no_grad():

        src_seq = torch.tensor(sequence, dtype=torch.long, device=mm.DEVICE).unsqueeze(0)
    
        num_tokens = len(sequence)
    
        tgt_a, tgt_b = gccn_dt_greedy_decode(model,  src_seq, max_len=num_tokens + translation_gap, bos_symbol=bos_symbol, pad_symbol=pad_symbol, eos_symbol=eos_symbol)

    return tgt_a, tgt_b

#  Realiza inferencia utilizando modelos Seq2Seq Transformer.

#    Parámetros:
#        tokenized_subset (list): Conjunto de datos tokenizados para la inferencia.
#        tst_models (list, opcional): Lista de modelos de Transformer a utilizar para la inferencia.
#        tst_slangs (list, opcional): Lista de lenguajes fuente de los datos.
#        tst_tlangs (list, opcional): Lista de lenguajes objetivo para la traducción.
#        threshold (float, opcional): Umbral BLEU para imprimir resultados.
#        limit (int, opcional): Límite en el número de muestras para la inferencia.
#        samples (list, opcional): Índices de las muestras específicas para la inferencia.

#    Retorna:
#        dict, dict: Métricas de los modelos y conjunto de muestras inferidas.

def run_tfrmr_st_inference(tokenized_subset, tst_models = [], tst_slangs = [mc.Language.EN, mc.Language.EN], tst_tlangs = [mc.Language.ES, mc.Language.FR], threshold = 0.5, limit = 0, samples = []):
    

    # Inicializa algunas variables necesarias, como diccionarios para almacenar las métricas 
    # del modelo y las muestras inferidas, así como otras variables para el control del flujo
    metrics_by_model = {}
    sample_set = {}
    max_printing_results = 10
    m_count = 0
    collection_of_samples = tokenized_subset
    # Verifica si hay muestras específicas para la inferencia. Si se proporcionan muestras, 
    # selecciona esas muestras del conjunto de datos tokenizado.

    if len(samples) > 0:
        collection_of_samples = []
        for index in samples:
            collection_of_samples.append(tokenized_subset[index])

    sample_set_index = 0

    for m, s_lang, t_lang in zip(tst_models, tst_slangs, tst_tlangs):
        print ("Model: {0}".format(m.getName()))
        i = 0

        local_blue_score = 0.0
        local_blue_count = 0
        cool_results = 0
        inference_ttl_time = 0

        # Itera sobre cada muestra en el conjunto de muestras (o en el conjunto completo si no se 
        # proporcionan muestras específicas).
        for (src, _, _, src_EN_tokens, tgt_ES_tokens, tgt_FR_tokens) in iter(collection_of_samples):


            # Realiza la traducción de la secuencia de entrada utilizando el modelo y 
            # el lenguaje objetivo actual.

            trn_inf_s_time = timer()
            x = tfrmr_st_translate_seq(m, src, t_lang)
            trn_inf_e_time = timer()
            trn_inf_time = trn_inf_e_time - trn_inf_s_time

            inference_ttl_time += trn_inf_time

            candidates = None
            references = None
            if t_lang == mc.Language.ES:
                references = tgt_ES_tokens
                candidates = mc.es_vocabulary.lookup_tokens(list(x.cpu().numpy()))
               
            if t_lang == mc.Language.FR:
                references = tgt_FR_tokens               
                candidates = mc.fr_vocabulary.lookup_tokens(list(x.cpu().numpy()))

            candidates = clean_seq_of_tokens(candidates)
            references = clean_seq_of_tokens(references)

            # Calcula el puntaje BLEU comparando la traducción generada con la referencia (texto de destino).
            BLEU = sentence_bleu([references], candidates, weights=(0.25, 0.25, 0, 0))

            # Si se proporcionaron muestras específicas, almacena información sobre cada muestra, como el texto 
            # de origen, la traducción de referencia, la traducción generada y el puntaje BLEU.

            if (len(samples) > 0):

                src_txt = " ".join(src_EN_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                ground_truth = " ".join(references).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                pred_txt = " ".join(candidates).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")
                    
                sample_set[sample_set_index + 1] = dict(
                model_name = m.getName(),
                SRC_LNG = 'EN',
                SRC_TXT = src_txt,
                TGT_LNG = t_lang,
                TGT_GTV = ground_truth,
                TGT_OUT = pred_txt,               
                TGT_BLEU = BLEU
                )

                sample_set_index += 1

            local_blue_score += BLEU
            local_blue_count += 1

            # Imprime información sobre las muestras cuyo puntaje BLEU supera el umbral especificado.
            if (cool_results < max_printing_results):
                if BLEU > threshold:

                    src_txt = " ".join(src_EN_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                    ground_truth = " ".join(references).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                    pred_txt = " ".join(candidates).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")
                                       
                    print("{0}.SRC_SEQ: {1}".format(i+1, src_txt))
                    print("{0}.TGT_SEQ: {1}".format(i+1, ground_truth))
                    print("{0}.OUT_SEQ: {1}".format(i+1, pred_txt))
                    print("{0}.BLUE: {1}".format(i+1, BLEU))
                    cool_results += 1

            # Calcula y almacena el tiempo promedio de inferencia y el puntaje BLEU promedio del modelo.
            if i % 1000 == 999:           
                now = datetime.datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                print(' [{0}, {1}] - inference - {2} BLEU: {3}'.format(time_str, m.getName(), i, local_blue_score / i ))
                
            i+=1

            if (limit > 0):
                if (i > limit):
                    break

        inference_avg_time = inference_ttl_time / i    
        
        local_blue_score = local_blue_score / local_blue_count

        print("BLEU Score for {0} = {1}".format(m.getName(), local_blue_score))

        this_lang = "EN"

        if t_lang == mc.Language.ES:
            this_lang = "ES" 
        if t_lang == mc.Language.FR:
            this_lang = "FR" 

        metrics_by_model[m_count + 1] = dict(
                model_name = m.getName(),
                model_lang = this_lang,
                BLEU = local_blue_score,
                inference_avg_time = inference_avg_time,
                inference_ttl_time = inference_ttl_time
            )
        
        m_count +=1
        
    return metrics_by_model, sample_set

# Esta función run_tfrmr_dt_inference lleva a cabo la inferencia utilizando modelos de traducción 
# doble (DoubleTaskSeq2SeqTransformer) sobre un conjunto de datos tokenizado. Aquí está la documentación 
# detallada de la función:
# Parámetros:
#   tokenized_subset: Conjunto de datos tokenizado que contiene las muestras a traducir.
#   tst_models: Lista de modelos de traducción doble a utilizar para la inferencia.
#   threshold: Umbral para considerar una traducción válida basada en el puntaje BLEU.
#   limit: Límite opcional para el número máximo de muestras a inferir.
#   samples: Lista opcional de índices de muestras específicas a inferir.
# Salida:
#   metrics_by_model: Diccionario que contiene métricas de rendimiento por modelo, incluyendo el puntaje BLEU promedio y el tiempo promedio de inferencia.
#   sample_set: Diccionario que almacena información sobre las muestras inferidas, incluyendo el texto de origen, las traducciones de destino (ES y FR), 
#               los puntajes BLEU correspondientes, y las traducciones generadas.

def run_tfrmr_dt_inference(tokenized_subset, tst_models = [], threshold = 0.5, limit = 0, samples = []):
    # La función inicializa variables y estructuras de datos necesarias:
    max_printing_results = 10
    metrics_by_model = {}
    m_count = 0
    sample_set = {}
    collection_of_samples = tokenized_subset

    # Si se proporcionan muestras específicas, selecciona esas muestras del conjunto de datos tokenizado.
    if len(samples) > 0:
        collection_of_samples = []
        for index in samples:
            collection_of_samples.append(tokenized_subset[index])

    sample_set_index = 0

    # Itera sobre cada modelo en tst_models
    for m in tst_models:
        print ("Model: {0}".format(m.getName()))
        i = 0

        # Inicializa variables para calcular métricas de rendimiento del modelo, como el puntaje BLEU 
        # y el tiempo de inferencia.

        local_blue_es_score = 0.0
        local_blue_es_count = 0

        local_blue_fr_score = 0.0
        local_blue_fr_count = 0

        cool_results = 0

        inference_ttl_time = 0

        for (src, _, _, src_EN_tokens, tgt_ES_tokens, tgt_FR_tokens) in iter(collection_of_samples):
            # Realiza la traducción de la secuencia de entrada utilizando el modelo actual.
            
            trn_inf_s_time = timer()
            out_es, out_fr = tfrmr_dt_translate_seq(m, src)
            trn_inf_e_time = timer()

            trn_inf_time = trn_inf_e_time - trn_inf_s_time

            inference_ttl_time += trn_inf_time

               
            reference_es = tgt_ES_tokens
            candidate_es = mc.es_vocabulary.lookup_tokens(list(out_es.cpu().numpy()))
          
            reference_fr = tgt_FR_tokens
            candidate_fr = mc.fr_vocabulary.lookup_tokens(list(out_fr.cpu().numpy()))
          
            candidate_es = clean_seq_of_tokens(candidate_es)
            candidate_fr = clean_seq_of_tokens(candidate_fr)

            reference_es = clean_seq_of_tokens(reference_es)
            reference_fr = clean_seq_of_tokens(reference_fr)
           
            # Calcula el puntaje BLEU para las traducciones generadas en ambos idiomas de destino (ES y FR).
            
            BLEU_for_ES = sentence_bleu([reference_es], candidate_es, weights=(0.25, 0.25, 0, 0))
            BLEU_for_FR = sentence_bleu([reference_fr], candidate_fr, weights=(0.25, 0.25, 0, 0))

            local_blue_es_score += BLEU_for_ES
            local_blue_es_count += 1

            local_blue_fr_score += BLEU_for_FR
            local_blue_fr_count += 1

            # Almacena información sobre las muestras inferidas en sample_set, si es necesario.
            if (len(samples) > 0):

                src_txt = " ".join(src_EN_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                tgt_es_txt = " ".join(tgt_ES_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                tgt_fr_txt = " ".join(tgt_FR_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                pred_txt_es = " ".join(candidate_es).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")
                pred_txt_fr = " ".join(candidate_fr).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")
                    
                sample_set[sample_set_index + 1] = dict(
                model_name = m.getName(),

                SRC_LNG = 'EN',
                SRC_TXT = src_txt,
                TGT_A_LNG = 'ES',
                TGT_A_GTV = tgt_es_txt,
                TGT_A_OUT = pred_txt_es,               
                TGT_A_BLEU = BLEU_for_ES,

                TGT_B_LNG = 'FR',
                TGT_B_GTV = tgt_fr_txt,
                TGT_B_OUT = pred_txt_fr,               
                TGT_B_BLEU = BLEU_for_FR

                )

                sample_set_index += 1

            # Imprime información sobre las muestras inferidas cuyo puntaje BLEU supera el umbral especificado.
            if (cool_results < max_printing_results):
                if BLEU_for_ES > threshold and BLEU_for_FR > threshold:
                    src_txt = " ".join(src_EN_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                    tgt_es_txt = " ".join(tgt_ES_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                    tgt_fr_txt = " ".join(tgt_FR_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                    pred_txt_es = " ".join(candidate_es).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")
                    pred_txt_fr = " ".join(candidate_fr).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")
                    print("{0}.SRC_EN_SEQ: {1}".format(i+1, src_txt))

                    print("{0}.TGT_ES_SEQ: {1}".format(i+1, tgt_es_txt))
                    print("{0}.OUT_ES_SEQ: {1}".format(i+1, pred_txt_es))

                    print("{0}.TGT_FR_SEQ: {1}".format(i+1, tgt_fr_txt))
                    print("{0}.OUT_ES_SEQ: {1}".format(i+1, pred_txt_fr))
                    print("{0}.BLEU_FR: {1}".format(i+1, BLEU_for_ES))
                    print("{0}.BLEU_ES: {1}".format(i+1, BLEU_for_FR))
                    cool_results += 1
            
            if i % 1000 == 999:           
                now = datetime.datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                print(' [{0}, {1}] - inference - {2} BLEU (ES): {3} BLEU (FR): {4}'.format(time_str, m.getName(), i, local_blue_es_score / i, local_blue_fr_score / i))
 
            i+=1

            if (limit > 0):
                if (i > limit):
                    break
        # Actualiza las métricas del modelo con los puntajes BLEU y el tiempo de inferencia.           
        local_blue_es_score = local_blue_es_score / local_blue_es_count
        local_blue_fr_score = local_blue_fr_score / local_blue_fr_count

        inference_avg_time = inference_ttl_time / i

        print("BLEU (ES) Score for {0} = {1}".format(m.getName(), local_blue_es_score))
        print("BLEU (FR) Score for {0} = {1}".format(m.getName(), local_blue_fr_score))

        # Calcula y almacena el puntaje BLEU promedio y el tiempo de inferencia promedio para cada modelo.
        metrics_by_model[m_count + 1] = dict(
                model_name = m.getName(),
                model_lang = "ES/FR",
                BLEU_ES = local_blue_es_score,
                BLEU_FR = local_blue_fr_score,
                inference_avg_time = inference_avg_time,
                inference_ttl_time = inference_ttl_time
            )
        m_count +=1
    # Retorna las métricas de rendimiento por modelo y el conjunto de muestras inferidas.
    return metrics_by_model, sample_set



#    Realiza la inferencia utilizando modelos de traducción GCCN-Seq2Seq sobre un conjunto de datos tokenizado.

#    Parámetros:
#    - tokenized_subset: Conjunto de datos tokenizado que contiene las muestras a traducir.
#    - tst_models: Lista de modelos de traducción GCCN-Seq2Seq a utilizar para la inferencia.
#    - tst_slangs: Lista de idiomas de origen correspondientes a los modelos de traducción.
#    - tst_tlangs: Lista de idiomas de destino correspondientes a los modelos de traducción.
#    - threshold: Umbral para considerar una traducción válida basada en el puntaje BLEU.
#    - limit: Límite opcional para el número máximo de muestras a inferir.
#    - samples: Lista opcional de índices de muestras específicas a inferir.

#    Retorna:
#    - metrics_by_model: Diccionario que contiene métricas de rendimiento por modelo, incluyendo el puntaje BLEU promedio y el tiempo promedio de inferencia.
#    - sample_set: Diccionario que almacena información sobre las muestras inferidas, incluyendo el texto de origen, las traducciones de destino, los puntajes BLEU correspondientes, y las traducciones generadas.


def run_gccn_st_inference(tokenized_subset, tst_models = [], tst_slangs = [mc.Language.EN, mc.Language.EN], tst_tlangs = [mc.Language.ES, mc.Language.FR], threshold = 0.4, limit = 0, samples = []):
    max_printing_results = 10
    metrics_by_model = {}
    sample_set = {}
    m_count = 0
    collection_of_samples = tokenized_subset
    sample_set_index = 0

    if len(samples) > 0:
        collection_of_samples = []
        for index in samples:
            collection_of_samples.append(tokenized_subset[index])


    for m, s_lang, t_lang in zip(tst_models, tst_slangs, tst_tlangs):       
        print ("Model: {0}".format(m.getName()))
        i = 0
        local_blue_score = 0.0
        local_blue_count = 0
        cool_results = 0
        inference_ttl_time = 0

        for (src, _, _, src_EN_tokens, tgt_ES_tokens, tgt_FR_tokens) in iter(collection_of_samples):

            trn_inf_s_time = timer()
            x = gcnn_st_translate_seq(m, src, t_lang)
            trn_inf_e_time = timer()

            trn_inf_time = trn_inf_e_time - trn_inf_s_time
            inference_ttl_time += trn_inf_time
            
            candidate = None
            reference = None

            if t_lang == mc.Language.ES:
                reference = tgt_ES_tokens
                candidate = mc.es_vocabulary.lookup_tokens(x)

            if t_lang == mc.Language.FR:
                reference = tgt_FR_tokens
                candidate = mc.fr_vocabulary.lookup_tokens(x)

            candidate = clean_seq_of_tokens(candidate)
            reference = clean_seq_of_tokens(reference)
              
            BLEU = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0, 0))
            local_blue_score += BLEU
            local_blue_count += 1

            if (len(samples) > 0):
                src_txt = " ".join(src_EN_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                ground_truth = " ".join(reference).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                pred_txt = " ".join(candidate).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")
                    
                sample_set[sample_set_index + 1] = dict(
                model_name = m.getName(),
                SRC_LNG = 'EN',
                SRC_TXT = src_txt,
                TGT_LNG = t_lang,
                TGT_GTV = ground_truth,
                TGT_OUT = pred_txt,               
                TGT_BLEU = BLEU
                )

                sample_set_index += 1


            if (cool_results < max_printing_results):
                if BLEU > threshold:

                    src_txt = " ".join(src_EN_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                    ground_truth = " ".join(reference).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                    pred_txt = " ".join(candidate).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")

                    print("{0}.SRC_SEQ: {1}".format(i+1, src_txt))
                    print("{0}.TGT_SEQ: {1}".format(i+1, ground_truth))
                    print("{0}.OUT_SEQ: {1}".format(i+1, pred_txt))
                    print("{0}.BLUE: {1}".format(i+1, BLEU))
                    cool_results += 1
            
            if i % 1000 == 999:           
                now = datetime.datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                print(' [{0}, {1}] - inference - {2} BLEU: {3}'.format(time_str, m.getName(), i, local_blue_score / i))

            i+=1

            if (limit > 0):
                if (i > limit):
                    break
        
        inference_avg_time = inference_ttl_time / i

        local_blue_score = local_blue_score / local_blue_count

        print("BLEU Score for {0} = {1}".format(m.getName(), local_blue_score))

        this_lang = "EN"

        if t_lang == mc.Language.ES:
            this_lang = "ES" 
        if t_lang == mc.Language.FR:
            this_lang = "FR" 

        metrics_by_model[m_count + 1] = dict(
                model_name = m.getName(),
                model_lang = this_lang,
                BLEU = local_blue_score,
                inference_avg_time = inference_avg_time,
                inference_ttl_time = inference_ttl_time
            )
        
        m_count +=1
        
    return metrics_by_model, sample_set


#    Realiza la inferencia utilizando modelos de traducción GCCN-Dual-Target sobre un conjunto de datos tokenizado.

#    Parámetros:
#    - tokenized_subset: Conjunto de datos tokenizado que contiene las muestras a traducir.
#    - tst_models: Lista de modelos de traducción GCCN-Dual-Target a utilizar para la inferencia.
#    - threshold: Umbral para considerar una traducción válida basada en el puntaje BLEU.
#    - limit: Límite opcional para el número máximo de muestras a inferir.
#    - samples: Lista opcional de índices de muestras específicas a inferir.

#    Retorna:
#    - metrics_by_model: Diccionario que contiene métricas de rendimiento por modelo, incluyendo el puntaje BLEU para cada idioma de destino, y el tiempo promedio de inferencia.
#    - sample_set: Diccionario que almacena información sobre las muestras inferidas, incluyendo el texto de origen, las traducciones de destino para cada idioma, los puntajes BLEU correspondientes, y las traducciones generadas.

def run_gccn_dt_inference(tokenized_subset, tst_models = [], threshold = 0.4, limit = 0, samples = []):

    metrics_by_model = {} 
    max_printing_results = 10
    m_count = 0

    sample_set = {}

    collection_of_samples = tokenized_subset

    if len(samples) > 0:

        collection_of_samples = []

        for index in samples:
            collection_of_samples.append(tokenized_subset[index])


    sample_set_index = 0

    for m in tst_models:
        print ("Model: {0}".format(m.getName()))
        
        i = 0

        local_blue_es_score = 0.0
        local_blue_es_count = 0

        local_blue_fr_score = 0.0
        local_blue_fr_count = 0

        cool_results = 0

        inference_ttl_time = 0

        for (src, _, _, src_EN_tokens, tgt_ES_tokens, tgt_FR_tokens) in iter(collection_of_samples):
                   
            trn_inf_s_time = timer()
            out_ES, out_FR = gcnn_dt_translate_seq(m, src)
            trn_inf_e_time = timer()

            trn_inf_time = trn_inf_e_time - trn_inf_s_time
            inference_ttl_time += trn_inf_time

            reference_ES = tgt_ES_tokens
            reference_FR = tgt_FR_tokens

            canditate_ES = mc.es_vocabulary.lookup_tokens(out_ES)
            canditate_FR = mc.fr_vocabulary.lookup_tokens(out_FR)
            
            canditate_ES = clean_seq_of_tokens(canditate_ES)
            canditate_FR = clean_seq_of_tokens(canditate_FR)

            reference_ES = clean_seq_of_tokens(reference_ES)
            reference_FR = clean_seq_of_tokens(reference_FR)
                
            BLEU_ES = sentence_bleu([reference_ES], canditate_ES, weights=(0.25, 0.25, 0, 0))
            BLEU_FR = sentence_bleu([reference_FR], canditate_FR, weights=(0.25, 0.25, 0, 0))

            local_blue_es_score += BLEU_ES
            local_blue_es_count += 1

            local_blue_fr_score += BLEU_FR
            local_blue_fr_count += 1


            if (len(samples) > 0):

                src_txt = " ".join(src_EN_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                tgt_es_txt = " ".join(tgt_ES_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                tgt_fr_txt = " ".join(tgt_FR_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                pred_txt_es = " ".join(canditate_ES).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")
                pred_txt_fr = " ".join(canditate_FR).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")
                    
                sample_set[sample_set_index + 1] = dict(
                model_name = m.getName(),

                SRC_LNG = 'EN',
                SRC_TXT = src_txt,
                TGT_A_LNG = 'ES',
                TGT_A_GTV = tgt_es_txt,
                TGT_A_OUT = pred_txt_es,               
                TGT_A_BLEU = BLEU_ES,

                TGT_B_LNG = 'FR',
                TGT_B_GTV = tgt_fr_txt,
                TGT_B_OUT = pred_txt_fr,               
                TGT_B_BLEU = BLEU_FR)

                sample_set_index += 1


            if (cool_results < max_printing_results):
                if BLEU_ES > threshold and BLEU_FR > threshold:

                    src_txt = " ".join(src_EN_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")

                    tgt_es_txt = " ".join(tgt_ES_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")
                    tgt_fr_txt = " ".join(tgt_FR_tokens).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "")

                    pred_txt_ES = " ".join(canditate_ES).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")
                    pred_txt_FR = " ".join(canditate_FR).replace(mc.BOS_WORD, "").replace(mc.EOS_WORD, "").replace(mc.PAD_WORD, " ")

                    print("{0}.SRC_EN_SEQ: {1}".format(i+1, src_txt))

                    print("{0}.TGT_ES_SEQ: {1}".format(i+1, tgt_es_txt))
                    print("{0}.OUT_ES_SEQ: {1}".format(i+1, pred_txt_ES))

                    print("{0}.TGT_FR_SEQ: {1}".format(i+1, tgt_fr_txt))
                    print("{0}.OUT_ES_SEQ: {1}".format(i+1, pred_txt_FR))

                    print("{0}.BLEU_FR: {1}".format(i+1, BLEU_ES))
                    print("{0}.BLEU_ES: {1}".format(i+1, BLEU_FR))

                    cool_results += 1
            
            if i % 1000 == 999:           
                now = datetime.datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                print(' [{0}, {1}] - inference - {2} BLEU (ES): {3} BLEU (FR): {4}'.format(time_str, m.getName(), i, local_blue_es_score / i, local_blue_fr_score / i))

            i+=1

            if (limit > 0):
                if (i > limit):
                    break

        local_blue_es_score = local_blue_es_score / local_blue_es_count
        local_blue_fr_score = local_blue_fr_score / local_blue_fr_count

        inference_avg_time = inference_ttl_time / i

        print("BLEU (ES) Score for {0} = {1}".format(m.getName(), local_blue_es_score))
        print("BLEU (FR) Score for {0} = {1}".format(m.getName(), local_blue_fr_score))

        metrics_by_model[m_count + 1] = dict(
                model_name = m.getName(),
                model_lang = "ES/FR",
                BLEU_ES = local_blue_es_score,
                BLEU_FR = local_blue_fr_score,
                inference_avg_time = inference_avg_time,
                inference_ttl_time = inference_ttl_time
            )
        
        m_count +=1
    return metrics_by_model, sample_set
