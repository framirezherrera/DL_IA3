# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: dl_tfrmr_routines.py
# Descripción: Rutinas de entrenamiento para modelos basados en arquitecturas transformer para Seq-to-Seq
# Implementado por: Felipe Ramírez Herrera
# Curso Aprendizaje Profundo 1 y 2. Universidad de Valencia / ADEIT
# Ultima revisión: 11/04/2024 
# ----------------------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import torch
import torch.nn
import datetime
from timeit import default_timer as timer
import dl_common as mc
import dl_xxxx_models as mm

# Generación de máscaras para modelos basados en transformers.

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=mm.DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def get_padding_mask(src, padding_index = mc.PAD_EN_IDX):
    return (src == padding_index).transpose(0, 1)

def create_mask(src, tgt, src_padding_index = mc.PAD_EN_IDX, tgt_padding_index = mc.PAD_ES_IDX):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=mm.DEVICE).type(torch.bool)

    src_padding_mask = get_padding_mask(src, src_padding_index)
    tgt_padding_mask = get_padding_mask(tgt, tgt_padding_index)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def create_mask_for_dual_decoder(src, tgt_a, tgt_b, src_padding_index = mc.PAD_EN_IDX, tgt_a_padding_index = mc.PAD_ES_IDX, tgt_b_padding_index = mc.PAD_FR_IDX):
    src_seq_len = src.shape[0]
    tgt_a_seq_len = tgt_a.shape[0]
    tgt_b_seq_len = tgt_b.shape[0]

    tgt_a_mask = generate_square_subsequent_mask(tgt_a_seq_len)
    tgt_b_mask = generate_square_subsequent_mask(tgt_b_seq_len)

    src_mask = torch.zeros((src_seq_len, src_seq_len),device=mm.DEVICE).type(torch.bool)

    src_padding_mask = get_padding_mask(src, src_padding_index)
    
    tgt_a_padding_mask = get_padding_mask(tgt_a, tgt_a_padding_index)
    tgt_b_padding_mask = get_padding_mask(tgt_b, tgt_b_padding_index)

    return src_mask, tgt_a_mask, tgt_b_mask, src_padding_mask, tgt_a_padding_mask , tgt_b_padding_mask


# Pasos de entrenamiento para un single task model basado en Transformer Architecture

def train_single_transformer(model,  dataset, optimizer, loss_fn, src_pad_idx, tgt_pad_idx, device, clip=None, smoothing = True, gc_count = 2):
    model.train()       

    epoch_loss = 0
    epoch_accm = 0 
    epoch_wmem = 0
    epoch_lmem = 0    

    i = 0
    batches_before_gc = 0
       
    for  src, tgt in iter(dataset):

        tgt_input = tgt[:-1, :]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        epoch_wmem += mm.getSizeOf(src)
        epoch_wmem += mm.getSizeOf(tgt)
        epoch_wmem += mm.getSizeOf(src_mask)
        epoch_wmem += mm.getSizeOf(tgt_mask)
        epoch_wmem += mm.getSizeOf(src_padding_mask)
        epoch_wmem += mm.getSizeOf(tgt_padding_mask)

        optimizer.zero_grad()                    

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        epoch_lmem += mm.getSizeOf(logits)

        tgt_output = tgt[1:, :]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        loss.backward()

        if (not clip == None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step() 
                    
        accm = mm.masked_accuracy(tgt_output.reshape(-1), logits.reshape(-1, logits.shape[-1]),  tgt_pad_idx)           

        epoch_loss += loss
        epoch_accm += accm
        
        if i % 1000 == 999:           
            now = datetime.datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            print(' [{0}] - train - batch {1} loss: {2} acc: {3}'.format(time_str, i + 1, epoch_loss.item() / (i + 1), epoch_accm.item() / (i + 1) ))

        epoch_wmem += mm.getSizeOf(loss)
        epoch_wmem += mm.getSizeOf(accm)
        i += 1

        del src
        del tgt
        del src_mask
        del tgt_mask
        del src_padding_mask
        del tgt_padding_mask
        del loss
        del accm
        del logits

        if torch.cuda.is_available():
            batches_before_gc += 1
            if (batches_before_gc > gc_count):
                torch.cuda.empty_cache()       
                batches_before_gc = 0
        
    train_loss = (epoch_loss / mc.batches_for_training).item()  
    train_accm = (epoch_accm / mc.batches_for_training).item() 
    wrk_mem = epoch_wmem / mc.batches_for_training
    log_mem = epoch_lmem / mc.batches_for_training
    return train_loss, train_accm, wrk_mem, log_mem

# Pasos de evaluación para un single task model basado en Transformer Architecture

def evaluate_single_transformer(model,  dataset, loss_fn, src_pad_idx, tgt_pad_idx, device, gc_count = 2):
    model.eval()    
    epoch_loss = 0
    epoch_accm = 0 
    epoch_wmem = 0
    epoch_lmem = 0    
    with torch.no_grad():
  
        i = 0
        batches_before_gc = 0
        b_time = timer()
        for  src, tgt in iter(dataset):

            tgt_input = tgt[:-1, :]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            
            epoch_wmem += mm.getSizeOf(src)
            epoch_wmem += mm.getSizeOf(tgt)
            epoch_wmem += mm.getSizeOf(src_mask)
            epoch_wmem += mm.getSizeOf(tgt_mask)
            epoch_wmem += mm.getSizeOf(src_padding_mask)
            epoch_wmem += mm.getSizeOf(tgt_padding_mask)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            epoch_lmem += mm.getSizeOf(logits) # Contabiliza

            tgt_output = tgt[1:, :]
            
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            accm = mm.masked_accuracy(tgt_output.reshape(-1), logits.reshape(-1, logits.shape[-1]),  tgt_pad_idx)

            epoch_loss += loss
            epoch_accm += accm

            if i % 1000 == 999: 
                now = datetime.datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                print(' [{0}] - eval - batch {1} loss: {2} acc: {3}'.format(time_str, i + 1, epoch_loss.item() / (i + 1), epoch_accm.item() / (i + 1) ))

            epoch_wmem += mm.getSizeOf(loss)
            epoch_wmem += mm.getSizeOf(accm)

            i += 1

            del src
            del tgt
            del src_mask
            del src_padding_mask
            del tgt_mask
            del tgt_padding_mask
            del logits
            del loss
            del accm
            if torch.cuda.is_available():
                batches_before_gc += 1
                if (batches_before_gc > gc_count):
                    torch.cuda.empty_cache()       
                    batches_before_gc = 0
                    
    valid_loss = (epoch_loss / mc.batches_for_validation).item()
    valid_accm = (epoch_accm / mc.batches_for_validation).item()
    wrk_mem = epoch_wmem / mc.batches_for_validation
    log_mem = epoch_lmem / mc.batches_for_validation
    return valid_loss , valid_accm, wrk_mem, log_mem


# Ciclo de entrenamiento / validación para un single task model basado en Transformer Architecture

def run_single_transformer(model, filename, trnset, valset, optimizer, scheduler, loss_fn, src_pad_idx, tgt_pad_idx):
    print(model.getName())    
    
    metrics_by_epoch = {}
    
    best_valid_loss = float("inf")
    
    early_stopping_count = 0

    base_epoch = -1 
    
    # Check for previos training checkpoints (latest)
    if os.path.exists(filename.format(mc.NUMBER_OF_EPOCHS - 1)):
        base_epoch = mc.NUMBER_OF_EPOCHS - 1
    else:
        for epoch in range(mc.NUMBER_OF_EPOCHS):
            if os.path.exists(filename.format(epoch)):
                base_epoch = epoch
            else:
                break
    
    # Load the latest checkpoint 
    if (base_epoch >= 0):
        # Load model checkpoint file (latest)
        checkpoint = torch.load(filename.format(base_epoch))
        # Load model data only when match filename with epoch information
        if (checkpoint['epoch'] == base_epoch):
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            early_stopping_count = checkpoint['early_stopping_count']
            metrics_by_epoch = checkpoint['metrics_by_epoch']
            best_valid_loss = checkpoint['best_valid_loss']
            early_stopping_count = checkpoint['early_stopping_count']
            print("Continuing training model {0} from checkpoint at epoch {1}".format(model.getName(), base_epoch + 1))

            for i in range(1, base_epoch):
                print("- Epoch {0} achieves T.Loss = {1} E.Loss = {2}".format(i, metrics_by_epoch[i]['train_loss'], metrics_by_epoch[i]['valid_loss']))
            
            print("- best_valid_loss = {0}".format(best_valid_loss))

        else:
            raise RuntimeError("Checkpoint {0} is corrupt".format(base_epoch))

    if (base_epoch + 1 < mc.NUMBER_OF_EPOCHS):
        for epoch in range(base_epoch + 1, mc.NUMBER_OF_EPOCHS):

            trn_s_time = timer()      
            trn_loss, trn_accm, trn_wmem, trn_lmem = train_single_transformer(model, trnset, optimizer, loss_fn,  src_pad_idx, tgt_pad_idx, mm.DEVICE, clip=mc.CLIPPING_VALUE)
            trn_e_time = timer()
            trn_elapsed_time = trn_e_time - trn_s_time

            val_s_time = timer()      
            val_loss, val_accm, val_wmem, val_lmem = evaluate_single_transformer(model, valset, loss_fn, src_pad_idx, tgt_pad_idx, mm.DEVICE)
            val_e_time = timer()
            val_elapsed_time = val_e_time - val_s_time

            scheduler.step(val_loss)

            print("Model {0} at Epoch {1} Duration = {2} second(s) LR = {3}".format(model.getName(), epoch + 1, trn_elapsed_time + val_elapsed_time, scheduler.get_last_lr()))
            print("Model {0} at Epoch {1} Trn.Loss = {2} and Trn.Accuracy = {3}".format(model.getName(), epoch + 1, trn_loss, trn_accm))
            print("Model {0} at Epoch {1} Val.Loss = {2} and Val.Accuracy = {3}".format(model.getName(), epoch + 1, val_loss, val_accm))

            metrics_by_epoch[epoch + 1] = dict(
                model_name = model.getName(),
                model_epoch = epoch + 1,
                train_loss = trn_loss,
                train_accm = trn_accm,
                train_pplx = np.exp(trn_loss),
                train_wrkmem = trn_wmem,
                train_logmem = trn_lmem,
                valid_loss = val_loss,
                valid_accm = val_accm,
                valid_pplx = np.exp(val_loss),
                valid_wrkmem = val_wmem,
                valid_logmem = val_lmem,
                scheduler_lr = scheduler.get_last_lr(),
                trn_elapsed_time = trn_elapsed_time,
                val_elapsed_time = val_elapsed_time
            )

            temp_filename = "temp_" + filename.format(epoch)

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                early_stopping_count = 0
            elif epoch < mc.EARLY_STOPPING_EPOCHS:
                early_stopping_count += 1

            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'early_stopping_count': early_stopping_count,
            'metrics_by_epoch' : metrics_by_epoch,
            'best_valid_loss' : best_valid_loss,
            'early_stopping_count': early_stopping_count
            }, temp_filename)

            os.rename(temp_filename, filename.format(epoch))
                                 
            if early_stopping_count == mc.EARLY_STOPPING_EPOCHS:
                print("Early stopping triggered in epoch {0}".format(epoch + 1))
                break
        
            print("Waiting for gpu cooling time...")

            mm.let_gpu_rest(mc.INNER_GPU_REST_TIME)
        
    return metrics_by_epoch

# Pasos de entrenamiento para un dual task model basado en Transformer Architecture

def train_dual_task_transformer(model, dataset, optimizer, loss_fn_a, loss_fn_b, src_pad_idx, tgt_a_pad_idx, tgt_b_pad_idx, device, clip=None, smoothing = True, gc_count = 1):
    model.train()   


    epoch_loss_j = 0
    epoch_loss_a = 0
    epoch_loss_b = 0
    epoch_accm_a = 0 
    epoch_accm_b = 0 
    epoch_wmem = 0
    epoch_lmem = 0   

    step = 0
    batches_before_gc = 0
    for  src, tgt_a, tgt_b in iter(dataset):      

        tgt_a_input = tgt_a[:-1, :]
        tgt_b_input = tgt_b[:-1, :]
        
        src_mask, tgt_a_mask, tgt_b_mask, src_padding_mask, tgt_a_padding_mask, tgt_b_padding_mask  = create_mask_for_dual_decoder(src, tgt_a_input, tgt_b_input)

        epoch_wmem += mm.getSizeOf(src)
        epoch_wmem += mm.getSizeOf(tgt_a)
        epoch_wmem += mm.getSizeOf(tgt_b)
        epoch_wmem += mm.getSizeOf(src_mask)
        epoch_wmem += mm.getSizeOf(tgt_a_mask)
        epoch_wmem += mm.getSizeOf(tgt_b_mask)
        epoch_wmem += mm.getSizeOf(src_padding_mask)
        epoch_wmem += mm.getSizeOf(tgt_a_padding_mask)
        epoch_wmem += mm.getSizeOf(tgt_b_padding_mask)
        
        optimizer.zero_grad()   
                    
        logits_a, logits_b = model(src, tgt_a_input, tgt_b_input, src_mask, tgt_a_mask, tgt_b_mask, src_padding_mask, tgt_a_padding_mask, tgt_b_padding_mask, src_padding_mask)

        epoch_lmem += mm.getSizeOf(logits_a) 
        epoch_lmem += mm.getSizeOf(logits_b) 

        tgt_a_output = tgt_a[1:, :]
        tgt_b_output = tgt_b[1:, :]

        loss_a = loss_fn_a(logits_a.reshape(-1, logits_a.shape[-1]), tgt_a_output.reshape(-1))
        loss_b = loss_fn_b(logits_b.reshape(-1, logits_b.shape[-1]), tgt_b_output.reshape(-1))

        joint_loss = loss_a + loss_b

        joint_loss.backward()

        if (not clip == None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step() 

        epoch_loss_a += loss_a
        epoch_loss_b += loss_b

        epoch_loss_j += joint_loss           

        acc_a = mm.masked_accuracy(tgt_a_output.reshape(-1), logits_a.reshape(-1, logits_a.shape[-1]),  tgt_a_pad_idx)
        acc_b = mm.masked_accuracy(tgt_b_output.reshape(-1), logits_b.reshape(-1, logits_b.shape[-1]),  tgt_b_pad_idx)

        epoch_accm_a += acc_a
        epoch_accm_b += acc_b
        
        if step % 1000 == 999: 
            now = datetime.datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            print(' [{0}] - train - batch {1} loss (ES): {2} acc (ES): {3}'.format(time_str, step + 1, epoch_loss_a.item() / (step + 1), epoch_accm_a.item() / (step + 1) ))
            print(' [{0}] - train - batch {1} loss (FR): {2} acc (FR): {3}'.format(time_str, step + 1, epoch_loss_b.item() / (step + 1), epoch_accm_b.item() / (step + 1) ))


        epoch_wmem += mm.getSizeOf(acc_a)
        epoch_wmem += mm.getSizeOf(acc_b)
        epoch_wmem += mm.getSizeOf(loss_a)
        epoch_wmem += mm.getSizeOf(loss_b)
        epoch_wmem += mm.getSizeOf(joint_loss)

        step += 1

        
        del src
        del tgt_a
        del tgt_b

        del src_mask
        del src_padding_mask
        del tgt_a_mask
        del tgt_b_mask
        del tgt_a_padding_mask
        del tgt_b_padding_mask

        del acc_a
        del acc_b
        del loss_a
        del loss_b
        del joint_loss

        if torch.cuda.is_available():
            batches_before_gc += 1
            if (batches_before_gc > gc_count):
                torch.cuda.empty_cache()       
                batches_before_gc = 0

    train_loss_j = (epoch_loss_j / mc.batches_for_training).item()
    train_loss_a = (epoch_loss_a / mc.batches_for_training).item()  
    train_loss_b = (epoch_loss_b / mc.batches_for_training).item()  
    train_accm_a = (epoch_accm_a / mc.batches_for_training).item() 
    train_accm_b = (epoch_accm_b / mc.batches_for_training).item() 
    wrk_mem = epoch_wmem / mc.batches_for_training
    log_mem = epoch_lmem / mc.batches_for_training
    return train_loss_j, train_loss_a, train_loss_b, train_accm_a, train_accm_b, wrk_mem, log_mem

# Pasos de evaluación para un dual task model basado en Transformer Architecture

def evaluate_dual_task_transformer(model,  dataset, loss_fn_a, loss_fn_b, src_pad_idx, tgt_a_pad_idx , tgt_b_pad_idx, device, gc_count = 1):
    model.eval()    
    epoch_loss_j = 0
    epoch_loss_a = 0
    epoch_loss_b = 0
    epoch_accm_a = 0 
    epoch_accm_b = 0 
    epoch_wmem = 0
    epoch_lmem = 0 
    with torch.no_grad():
        step = 0
        batches_before_gc = 0
        for  src, tgt_a, tgt_b in iter(dataset):
            tgt_a_input = tgt_a[:-1, :]
            tgt_b_input = tgt_b[:-1, :]
            src_mask, tgt_a_mask, tgt_b_mask, src_padding_mask, tgt_a_padding_mask , tgt_b_padding_mask = create_mask_for_dual_decoder(src, tgt_a_input, tgt_b_input)
            epoch_wmem += mm.getSizeOf(src)
            epoch_wmem += mm.getSizeOf(tgt_a)
            epoch_wmem += mm.getSizeOf(tgt_b)
            epoch_wmem += mm.getSizeOf(src_mask)
            epoch_wmem += mm.getSizeOf(tgt_a_mask)
            epoch_wmem += mm.getSizeOf(tgt_b_mask)
            epoch_wmem += mm.getSizeOf(src_padding_mask)
            epoch_wmem += mm.getSizeOf(tgt_a_padding_mask)
            epoch_wmem += mm.getSizeOf(tgt_b_padding_mask)
            logits_a, logits_b = model(src, tgt_a_input, tgt_b_input, src_mask, tgt_a_mask, tgt_b_mask, src_padding_mask, tgt_a_padding_mask, tgt_b_padding_mask, src_padding_mask)
            epoch_lmem += mm.getSizeOf(logits_a) 
            epoch_lmem += mm.getSizeOf(logits_b) 
            tgt_a_output = tgt_a[1:, :]
            tgt_b_output = tgt_b[1:, :]
            loss_a = loss_fn_a(logits_a.reshape(-1, logits_a.shape[-1]), tgt_a_output.reshape(-1))
            loss_b = loss_fn_b(logits_b.reshape(-1, logits_b.shape[-1]), tgt_b_output.reshape(-1))
            joint_loss = loss_a + loss_b
            epoch_loss_j += joint_loss
            epoch_loss_a += loss_a
            epoch_loss_b += loss_b
            acc_a = mm.masked_accuracy(tgt_a_output.reshape(-1), logits_a.reshape(-1, logits_a.shape[-1]),  tgt_a_pad_idx)
            acc_b = mm.masked_accuracy(tgt_b_output.reshape(-1), logits_b.reshape(-1, logits_b.shape[-1]),  tgt_b_pad_idx)
            epoch_accm_a += acc_a
            epoch_accm_b += acc_b
            if step % 1000 == 999: 
                now = datetime.datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                print(' [{0}] - eval - batch {1} loss (ES): {2} acc (ES): {3}'.format(time_str, step + 1, epoch_loss_a.item() / (step + 1), epoch_accm_a.item() / (step + 1) ))
                print(' [{0}] - eval - batch {1} loss (FR): {2} acc (FR): {3}'.format(time_str, step + 1, epoch_loss_b.item() / (step + 1), epoch_accm_b.item() / (step + 1) ))
            epoch_wmem += mm.getSizeOf(acc_a)
            epoch_wmem += mm.getSizeOf(acc_b)
            epoch_wmem += mm.getSizeOf(loss_a)
            epoch_wmem += mm.getSizeOf(loss_b)
            epoch_wmem += mm.getSizeOf(joint_loss)
            step += 1

            del src
            del tgt_a
            del tgt_b
            del src_mask
            del tgt_a_mask
            del tgt_b_mask
            del src_padding_mask
            del tgt_a_padding_mask
            del tgt_b_padding_mask
            del logits_a
            del logits_b
            del loss_a
            del loss_b
            del acc_a
            del acc_b
            del joint_loss

            if torch.cuda.is_available():
                batches_before_gc += 1
                if (batches_before_gc > gc_count):
                    torch.cuda.empty_cache()       
                    batches_before_gc = 0
    
    result_loss_j = (epoch_loss_j / mc.batches_for_validation).item()
    result_loss_a = (epoch_loss_a / mc.batches_for_validation).item()  
    result_loss_b = (epoch_loss_b / mc.batches_for_validation).item()  
    result_accm_a = (epoch_accm_a / mc.batches_for_validation).item() 
    resukt_accm_b = (epoch_accm_b / mc.batches_for_validation).item() 
    wrk_mem = epoch_wmem / mc.batches_for_validation
    log_mem = epoch_lmem / mc.batches_for_validation
    return result_loss_j, result_loss_a, result_loss_b, result_accm_a, resukt_accm_b, wrk_mem, log_mem

# Ciclo completo de entrenamiento / validación para un dual task model basado en Transformer Architecture

def run_dual_transformer(model, filename, trnset, valset, optimizer, scheduler, loss_fn_a, loss_fn_b, src_pad_idx, tgt_a_pad_idx , tgt_b_pad_idx):   
    print(model.getName())
    
    metrics_by_epoch = {}
    best_valid_loss_a = float("inf")
    best_valid_loss_b = float("inf")
    early_stopping_count = 0

    base_epoch = -1 

    # Check for previos training checkpoints (latest)
    if os.path.exists(filename.format(mc.NUMBER_OF_EPOCHS - 1)):
        base_epoch = mc.NUMBER_OF_EPOCHS - 1
    else:
        for epoch in range(mc.NUMBER_OF_EPOCHS):
            if os.path.exists(filename.format(epoch)):
                base_epoch = epoch
            else:
                break
    
    # Load the latest checkpoint 
    if (base_epoch >= 0):
        # Load model checkpoint file (latest)
        checkpoint = torch.load(filename.format(base_epoch))
        # Load model data only when match filename with epoch information
        if (checkpoint['epoch'] == base_epoch):
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            early_stopping_count = checkpoint['early_stopping_count']
            metrics_by_epoch = checkpoint['metrics_by_epoch']
            best_valid_loss_a = checkpoint['best_valid_loss_a']
            best_valid_loss_b = checkpoint['best_valid_loss_b']
            print("Continuing training model {0} from checkpoint at epoch {1}".format(model.getName(), base_epoch + 1))

            for i in range(1, base_epoch):
                print("- Epoch {0} achieves T.Loss(ES) = {1} E.Loss(ES) = {2}".format(i, metrics_by_epoch[i]['train_loss_a'], metrics_by_epoch[i]['valid_loss_a']))
                print("- Epoch {0} achieves T.Loss(FR) = {1} E.Loss(FR) = {2}".format(i, metrics_by_epoch[i]['train_loss_b'], metrics_by_epoch[i]['valid_loss_b']))
            print("- best_valid_loss (ES) = {0}".format(best_valid_loss_a))
            print("- best_valid_loss (FR) = {0}".format(best_valid_loss_b))

        else:
            raise RuntimeError("Checkpoint {0} is corrupt".format(base_epoch))
    if (base_epoch + 1 < mc.NUMBER_OF_EPOCHS):
        for epoch in range(base_epoch + 1, mc.NUMBER_OF_EPOCHS):

            trn_s_time = timer()    
            train_joint_loss, train_loss_a, train_loss_b, train_acc_a, train_acc_b, trn_wrk_mem, trn_log_mem = train_dual_task_transformer(model, trnset, optimizer, loss_fn_a, loss_fn_b,  src_pad_idx, tgt_a_pad_idx, tgt_b_pad_idx, mm.DEVICE, clip=mc.CLIPPING_VALUE)
            trn_e_time = timer()
            trn_elapsed_time = trn_e_time - trn_s_time

            val_s_time = timer()
            valid_joint_loss, valid_loss_a, valid_loss_b, valid_acc_a, valid_acc_b, val_wrk_mem, val_log_mem = evaluate_dual_task_transformer(model, valset, loss_fn_a, loss_fn_b, src_pad_idx, tgt_a_pad_idx, tgt_b_pad_idx, mm.DEVICE)
            val_e_time = timer()
            val_elapsed_time = val_e_time - val_s_time

            scheduler.step(valid_joint_loss)

            print("Model {0} at Epoch {1} Duration = {2} second(s)".format(model.getName(), epoch + 1, trn_elapsed_time + val_elapsed_time))
            print("Model {0} at Epoch {1} Trn(ES).Loss = {2} Trn(FR).Loss = {3} and Trn(ES).Accuracy = {4} and Trn(FR).Accuracy = {5}".format(model.getName(), epoch + 1, train_loss_a, train_loss_b, train_acc_a, train_acc_b))
            print("Model {0} at Epoch {1} Val(ES).Loss = {2} Val(FR).Loss = {3} and Val(ES).Accuracy = {4} and Val(FR).Accuracy = {5}".format(model.getName(), epoch + 1, valid_loss_a, valid_loss_b, valid_acc_a, valid_acc_b))

            metrics_by_epoch[epoch + 1] = dict(
                model_name = model.getName(),
                model_epoch = epoch + 1,
                train_joint_loss = train_joint_loss,
                train_joint_accm = 0.5 * (train_acc_a + train_acc_b),
                train_loss_a = train_loss_a,
                train_loss_b = train_loss_b,
                train_acc_a = train_acc_a,
                train_acc_b = train_acc_b,
                train_pplx = np.exp(train_joint_loss),
                train_wrkmem = trn_wrk_mem,
                train_logmem = trn_log_mem,
                valid_joint_loss = valid_joint_loss,
                valid_joint_accm = 0.5 * (valid_acc_a + valid_acc_b),
                valid_loss_a = valid_loss_a,
                valid_loss_b = valid_loss_b,
                valid_acc_a = valid_acc_a,
                valid_acc_b = valid_acc_b,
                valid_pplx = np.exp(valid_joint_loss),
                valid_wrkmem = val_wrk_mem,
                valid_logmem = val_log_mem,
                scheduler_lr = scheduler.get_last_lr(),
                trn_elapsed_time = trn_elapsed_time,
                val_elapsed_time = val_elapsed_time
            )

            temp_filename = "temp_" + filename.format(epoch)

            cond_a = valid_loss_a < best_valid_loss_a
            cond_b = valid_loss_b < best_valid_loss_b
            if cond_a or cond_b:
                best_valid_loss_a = valid_loss_a
                best_valid_loss_b = valid_loss_b
                early_stopping_count = 0
            elif epoch < mc.EARLY_STOPPING_EPOCHS:
                early_stopping_count += 1      

            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'early_stopping_count': early_stopping_count,
            'metrics_by_epoch' : metrics_by_epoch,
            'best_valid_loss_a' : valid_loss_a,
            'best_valid_loss_b' : valid_loss_b,
            }, temp_filename)

            os.rename(temp_filename, filename.format(epoch))
                             
            if early_stopping_count == mc.EARLY_STOPPING_EPOCHS:
                print("Early stopping triggered in epoch {0}".format(epoch + 1))
                break

            if (mc.INNER_GPU_REST_TIME != 0):
                print("Waiting for gpu cooling time...")
                mm.let_gpu_rest(mc.INNER_GPU_REST_TIME)

    return metrics_by_epoch
