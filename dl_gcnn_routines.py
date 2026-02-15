# ----------------------------------------------------------------------------------------------------------------------------------
# Archivo: dl_gcnn_routines.py
# Descripción: Rutinas de entrenamiento para modelos basados en arquitecturas Gated CNN para Seq-to-Seq
# Implementado por: Felipe Ramírez Herrera
# Curso Aprendizaje Profundo 1 y 2. Universidad de Valencia / ADEIT
# Ultima revisión: 11/04/2024 
# ----------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import dl_common as mc
import dl_xxxx_models as mm
import dl_gcnn_models as mg

import os

import time
import datetime

from timeit import default_timer as timer

# Pasos de entrenamiento para un single task model basado en Gated CNN Architecture

def train_single_cnn(model,  dataset, optimizer, loss_fn, src_pad_idx, tgt_pad_idx, device, clip=None, smoothing = True, gc_count = 10):
    model.train()       
    epoch_loss = 0
    epoch_accm = 0 
    epoch_wmem = 0
    epoch_lmem = 0   

    i = 0
    batches_before_gc = 0
    for  src, tgt in iter(dataset):

        src = src.permute(1, 0)
        tgt = tgt.permute(1, 0)       
        tgt_input = tgt[:, :-1]

        epoch_wmem += mm.getSizeOf(src)
        epoch_wmem += mm.getSizeOf(tgt)

        optimizer.zero_grad()                    

        logits, _ = model(src, tgt_input)

        epoch_lmem += mm.getSizeOf(logits)

        tgt_output = tgt[:,1:].contiguous().view(-1)

        log_output = logits.contiguous().view(-1, logits.shape[-1])

        loss = loss_fn(log_output, tgt_output)
                    
        loss.backward()

        if (not clip == None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step() 
            
        accm = mm.masked_accuracy(tgt_output, log_output, tgt_pad_idx)

        epoch_accm += accm
        epoch_loss += loss
        
        if i % 1000 == 999: 
            now = datetime.datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            print(' [{0}] - train - batch {1} loss: {2} acc: {3}'.format(time_str, i + 1, epoch_loss.item() / (i + 1), epoch_accm.item() / (i + 1) ))

        epoch_wmem += mm.getSizeOf(loss)
        epoch_wmem += mm.getSizeOf(accm)
        i += 1
    
        del src
        del tgt
        del tgt_input
        del tgt_output
        del loss
        del accm
        del logits
        batches_before_gc += 1

        if (batches_before_gc > gc_count):
            torch.cuda.empty_cache()       
            batches_before_gc = 0

    train_loss = (epoch_loss / mc.batches_for_training).item()
    train_accm = (epoch_accm / mc.batches_for_training).item() 
    wrk_mem = epoch_wmem / mc.batches_for_training
    log_mem = epoch_lmem / mc.batches_for_training    
    return train_loss, train_accm, wrk_mem, log_mem

# Pasos de evaluación para un single task model basado en Gated CNN Architecture

def eval_single_cnn(model,  dataset, loss_fn, src_pad_idx, tgt_pad_idx, device, gc_count = 10):
    model.eval()    
    epoch_loss = 0
    epoch_accm = 0 
    epoch_wmem = 0
    epoch_lmem = 0  
    with torch.no_grad():
        i = 0
        batches_before_gc = 0
        for  src, tgt in iter(dataset):

            src = src.permute(1, 0)
            tgt = tgt.permute(1, 0)
            
            tgt_input = tgt[:, :-1]

            epoch_wmem += mm.getSizeOf(src)
            epoch_wmem += mm.getSizeOf(tgt)

            logits, _ = model(src, tgt_input)

            epoch_lmem += mm.getSizeOf(logits)

            tgt_output = tgt[:,1:].contiguous().view(-1)
            log_output = logits.contiguous().view(-1, logits.shape[-1])

            loss = loss_fn(log_output, tgt_output)

            epoch_loss += loss
            
            accm = mm.masked_accuracy(tgt_output, log_output,  tgt_pad_idx)
            
            epoch_accm += accm
          
            if i % 1000 == 999: 
                now = datetime.datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                print(' [{0}] - tst - batch {1} loss: {2} acc: {3}'.format(time_str, i + 1, epoch_loss.item() / (i + 1), epoch_accm.item() / (i + 1) ))

            epoch_wmem += mm.getSizeOf(loss)
            epoch_wmem += mm.getSizeOf(accm)
            i += 1

            del src
            del tgt
            del tgt_input
            del tgt_output
            del logits
            del loss
            del accm
            batches_before_gc += 1
            if (batches_before_gc > gc_count):
                torch.cuda.empty_cache()       
                batches_before_gc = 0
    
    valid_loss = (epoch_loss / mc.batches_for_validation).item()
    valid_accm = (epoch_accm / mc.batches_for_validation).item()
    wrk_mem = epoch_wmem / mc.batches_for_validation
    log_mem = epoch_lmem / mc.batches_for_validation
    return valid_loss , valid_accm, wrk_mem, log_mem

# Época para un single task model basado en Gated CNN Architecture

def run_single_cnn(model, filename, trnset, valset, optimizer, scheduler, loss_fn, src_pad_idx, tgt_pad_idx):
    
    
    print(model.getName())    
    
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print('[{0}] - Training / Testing Process - Started!'.format(time_str))

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
            train_loss, train_accm, trn_wrk_mem, trn_log_mem = train_single_cnn(model, trnset, optimizer, loss_fn,  src_pad_idx, tgt_pad_idx, mm.DEVICE, clip=mc.CLIPPING_VALUE)
            trn_e_time = timer()
            trn_elapsed_time = trn_e_time - trn_s_time


            val_s_time = timer()  
            valid_loss, valid_accm, val_wrk_mem, val_log_mem = eval_single_cnn(model, valset, loss_fn, src_pad_idx, tgt_pad_idx, mm.DEVICE)
            val_e_time = timer()
            val_elapsed_time = val_e_time - val_s_time

            scheduler.step(valid_loss)


            print("Model {0} at Epoch {1} Duration = {2} second(s) LR = {3}".format(model.getName(), epoch + 1, trn_elapsed_time + val_elapsed_time, scheduler.get_last_lr()))
            print("Model {0} at Epoch {1} Trn.Loss = {2} and Trn.Accuracy = {3}".format(model.getName(), epoch + 1, train_loss, train_accm))
            print("Model {0} at Epoch {1} Val.Loss = {2} and Val.Accuracy = {3}".format(model.getName(), epoch + 1, valid_loss, valid_accm))

            metrics_by_epoch[epoch + 1] = dict(
                model_name = model.getName(),
                model_epoch = epoch + 1,
                train_loss = train_loss,
                train_accm = train_accm,
                train_pplx = np.exp(train_loss),
                train_wrkmem = trn_wrk_mem,
                train_logmem = trn_log_mem,
                valid_loss = valid_loss,
                valid_accm = valid_accm,
                valid_pplx = np.exp(valid_loss),
                valid_wrkmem = val_wrk_mem,
                valid_logmem = val_log_mem,
                scheduler_lr = scheduler.get_last_lr(),
                trn_elapsed_time = trn_elapsed_time,
                val_elapsed_time = val_elapsed_time
            )

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                early_stopping_count = 0
            elif epoch < mc.EARLY_STOPPING_EPOCHS:
                early_stopping_count += 1

            temp_filename = "temp_" + filename.format(epoch)

            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'early_stopping_count': early_stopping_count,
            'metrics_by_epoch' : metrics_by_epoch,
            'scheduler_state_dict' : scheduler.state_dict(),
            'best_valid_loss' : best_valid_loss
            }, temp_filename)

            os.rename(temp_filename, filename.format(epoch))
                                 
            if early_stopping_count == mc.EARLY_STOPPING_EPOCHS:
                print("Early stopping triggered in epoch {0}".format(epoch + 1))
                break
        
            print("Waiting for gpu cooling time...")

            mm.let_gpu_rest(mc.INNER_GPU_REST_TIME)

    return metrics_by_epoch

# Pasos de entrenamiento para un double task model basado en Gated CNN Architecture

def train_dual_cnn(model : mg.GatedConvDualTaskSeq2Seq,  dataset, optimizer, loss_a_fn, loss_b_fn, src_pad_idx, tgt_a_pad_idx, tgt_b_pad_idx, device, clip=None, smoothing = True, gc_count = 10):
    model.train()       
    epoch_loss_a = 0
    epoch_loss_b = 0
    epoch_loss_t = 0
    epoch_accm_a = 0 
    epoch_accm_b = 0 
    epoch_wmem = 0
    epoch_lmem = 0   

    i = 0
    batches_before_gc = 0
    for  src, tgt_a, tgt_b in iter(dataset):

        src = src.permute(1, 0)
        tgt_a = tgt_a.permute(1, 0)   
        tgt_b = tgt_b.permute(1, 0)       

        tgt_a_input = tgt_a[:, :-1]
        tgt_b_input = tgt_b[:, :-1]

        epoch_wmem += mm.getSizeOf(src)
        epoch_wmem += mm.getSizeOf(tgt_a)
        epoch_wmem += mm.getSizeOf(tgt_b)

        optimizer.zero_grad()                    

        logits_a, logits_b = model(src, tgt_a_input, tgt_b_input)

        epoch_lmem += mm.getSizeOf(logits_a)
        epoch_lmem += mm.getSizeOf(logits_b)

        tgt_a_output = tgt_a[:,1:].contiguous().view(-1)
        tgt_b_output = tgt_b[:,1:].contiguous().view(-1)

        log_a_output = logits_a.contiguous().view(-1, logits_a.shape[-1])
        log_b_output = logits_b.contiguous().view(-1, logits_b.shape[-1])

        loss_for_a = loss_a_fn(log_a_output, tgt_a_output)
        loss_for_b = loss_b_fn(log_b_output, tgt_b_output)

        loss = loss_for_a + loss_for_b
                    
        loss.backward()

        if (not clip == None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step() 
            
        accm_for_a = mm.masked_accuracy(tgt_a_output, log_a_output, tgt_a_pad_idx)
        accm_for_b = mm.masked_accuracy(tgt_b_output, log_b_output, tgt_b_pad_idx)

        epoch_accm_a += accm_for_a
        epoch_accm_b += accm_for_b
        epoch_loss_a += loss_for_a
        epoch_loss_b += loss_for_b
        epoch_loss_t += loss
        if i % 1000 == 999: 
            now = datetime.datetime.now()
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            print(' [{0}] - train - batch {1} loss (ES): {2} acc (ES): {3}'.format(time_str, i + 1, epoch_loss_a.item() / (i + 1), epoch_accm_a.item() / (i + 1) ))
            print(' [{0}] - train - batch {1} loss (FR): {2} acc (FR): {3}'.format(time_str, i + 1, epoch_loss_b.item() / (i + 1), epoch_accm_b.item() / (i + 1) ))

        epoch_wmem += mm.getSizeOf(loss)
        epoch_wmem += mm.getSizeOf(loss_for_a)
        epoch_wmem += mm.getSizeOf(loss_for_b)
        epoch_wmem += mm.getSizeOf(accm_for_a)
        epoch_wmem += mm.getSizeOf(accm_for_b)
        i += 1
    
        del src
        del tgt_a
        del tgt_b
        del tgt_a_input
        del tgt_b_input
        del tgt_a_output
        del tgt_b_output
        del loss
        del loss_for_a
        del loss_for_b
        del accm_for_a
        del accm_for_b
        del logits_a
        del logits_b

        if torch.cuda.is_available():
            batches_before_gc += 1
            if (batches_before_gc > gc_count):
                torch.cuda.empty_cache()       
                batches_before_gc = 0

    result_loss_t = (epoch_loss_t / mc.batches_for_training).item()
    result_loss_a = (epoch_loss_a / mc.batches_for_training).item()  
    result_loss_b = (epoch_loss_b / mc.batches_for_training).item()  
    result_accm_a = (epoch_accm_a / mc.batches_for_training).item() 
    resukt_accm_b = (epoch_accm_b / mc.batches_for_training).item() 
    wrk_mem = epoch_wmem / mc.batches_for_training
    log_mem = epoch_lmem / mc.batches_for_training
    return result_loss_t, result_loss_a, result_loss_b, result_accm_a, resukt_accm_b, wrk_mem, log_mem

# Pasos de evaluación para un double task model basado en Gated CNN Architecture

def eval_dual_cnn(model : mg.GatedConvDualTaskSeq2Seq,  dataset, loss_a_fn, loss_b_fn, src_pad_idx, tgt_a_pad_idx, tgt_b_pad_idx, device, gc_count = 10):
    model.eval()    
    epoch_loss_t = 0
    epoch_loss_a = 0
    epoch_loss_b = 0
    epoch_accm_a = 0 
    epoch_accm_b = 0 
    epoch_wmem = 0
    epoch_lmem = 0  
    with torch.no_grad():
        i = 0
        batches_before_gc = 0
        for  src, tgt_a, tgt_b in iter(dataset):

            src = src.permute(1, 0)
            tgt_a = tgt_a.permute(1, 0)
            tgt_b = tgt_b.permute(1, 0)
            
            tgt_a_input = tgt_a[:, :-1]
            tgt_b_input = tgt_b[:, :-1]

            epoch_wmem += mm.getSizeOf(src)
            epoch_wmem += mm.getSizeOf(tgt_a)
            epoch_wmem += mm.getSizeOf(tgt_b)

            logits_a, logits_b = model(src, tgt_a_input, tgt_b_input)

            epoch_lmem += mm.getSizeOf(logits_a)
            epoch_lmem += mm.getSizeOf(logits_b)

            tgt_a_output = tgt_a[:,1:].contiguous().view(-1)
            tgt_b_output = tgt_b[:,1:].contiguous().view(-1)

            log_a_output = logits_a.contiguous().view(-1, logits_a.shape[-1])
            log_b_output = logits_b.contiguous().view(-1, logits_b.shape[-1])

            loss_for_a = loss_a_fn(log_a_output, tgt_a_output)
            loss_for_b = loss_b_fn(log_b_output, tgt_b_output)
            epoch_loss_a += loss_for_a
            epoch_loss_b += loss_for_b
            epoch_loss_t += loss_for_a + loss_for_b
            
            accm_for_a = mm.masked_accuracy(tgt_a_output, log_a_output, tgt_a_pad_idx)
            accm_for_b = mm.masked_accuracy(tgt_b_output, log_b_output, tgt_b_pad_idx)
            
            epoch_accm_a += accm_for_a
            epoch_accm_b += accm_for_b

            if i % 1000 == 999: 
                now = datetime.datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                print(' [{0}] - eval - batch {1} loss (ES): {2} acc (ES): {3}'.format(time_str, i + 1, epoch_loss_a.item() / (i + 1), epoch_accm_a.item() / (i + 1) ))
                print(' [{0}] - eval - batch {1} loss (FR): {2} acc (FR): {3}'.format(time_str, i + 1, epoch_loss_b.item() / (i + 1), epoch_accm_b.item() / (i + 1) ))

            epoch_wmem += mm.getSizeOf(loss_for_a)
            epoch_wmem += mm.getSizeOf(loss_for_b)
            epoch_wmem += mm.getSizeOf(accm_for_a)
            epoch_wmem += mm.getSizeOf(accm_for_b)

            i += 1

            del src
            del tgt_a
            del tgt_b
            del tgt_a_input
            del tgt_b_input
            del tgt_a_output
            del tgt_b_output
            del logits_a
            del logits_b
            del loss_for_a
            del loss_for_b
            del accm_for_a
            del accm_for_b
            if torch.cuda.is_available():
                batches_before_gc += 1
                if (batches_before_gc > gc_count):
                    torch.cuda.empty_cache()       
                    batches_before_gc = 0
    
    result_loss_t = (epoch_loss_t / mc.batches_for_validation).item()
    result_loss_a = (epoch_loss_a / mc.batches_for_validation).item()  
    result_loss_b = (epoch_loss_b / mc.batches_for_validation).item()  
    result_accm_a = (epoch_accm_a / mc.batches_for_validation).item() 
    resukt_accm_b = (epoch_accm_b / mc.batches_for_validation).item() 
    wrk_mem = epoch_wmem / mc.batches_for_validation
    log_mem = epoch_lmem / mc.batches_for_validation
    return result_loss_t, result_loss_a, result_loss_b, result_accm_a, resukt_accm_b, wrk_mem, log_mem

# Epoca para un double task model basado en Gated CNN Architecture

def run_dual_cnn(model : mg.GatedConvDualTaskSeq2Seq, filename, trnset, valset, optimizer, scheduler, loss_fn_a, loss_fn_b, src_pad_idx, tgt_a_pad_idx , tgt_b_pad_idx):   
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
            train_joint_loss, train_loss_a, train_loss_b, train_acc_a, train_acc_b, trn_wrk_mem, trn_log_mem = train_dual_cnn(model, trnset, optimizer, loss_fn_a, loss_fn_b,  src_pad_idx, tgt_a_pad_idx, tgt_b_pad_idx, mm.DEVICE, clip=mc.CLIPPING_VALUE)
            trn_e_time = timer()
            trn_elapsed_time = trn_e_time - trn_s_time

            val_s_time = timer()
            valid_joint_loss, valid_loss_a, valid_loss_b, valid_acc_a, valid_acc_b, val_wrk_mem, val_log_mem = eval_dual_cnn(model, valset, loss_fn_a, loss_fn_b, src_pad_idx, tgt_a_pad_idx, tgt_b_pad_idx, mm.DEVICE)
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

            cond_a = valid_loss_a < best_valid_loss_a
            cond_b = valid_loss_b < best_valid_loss_b
            if cond_a or cond_b:
                best_valid_loss_a = valid_loss_a
                best_valid_loss_b = valid_loss_b
                early_stopping_count = 0
            elif epoch < mc.EARLY_STOPPING_EPOCHS:
                early_stopping_count += 1   

            temp_filename = "temp_" + filename.format(epoch)

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