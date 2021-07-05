"""
File used to define useful utility functions during training. Mainly based on [GitHub repository](https://github.com/intersun/PKD-for-BERT-Model-Compression) for [Patient Knowledge Distillation for BERT Model Compression](https://arxiv.org/abs/1908.09355).
"""
import logging
import torch
import os

import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from torch import nn
from tqdm import tqdm

from utils.nli_data_processing import compute_metrics


logger = logging.getLogger(__name__)


def fill_tensor(tensor, batch_size):
    """
    for DataDistributed problem in pytorch  ...
    :param tensor:
    :param batch_size:
    :return:
    """
    if len(tensor) % batch_size != 0:
        diff = batch_size - len(tensor) % batch_size
        tensor += tensor[:diff]
    return tensor


def count_parameters(model, trainable_only=True, is_dict=False):
    if is_dict:
        return sum(np.prod(list(model[k].size())) for k in model)
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def load_model(model, checkpoint, args, mode='exact', train_mode='finetune', verbose=True, DEBUG=False):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """

    n_gpu = args.n_gpu
    device = args.device
    local_rank = -1
    if checkpoint in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint))
        model_state_dict = torch.load(checkpoint)
        old_keys = []
        new_keys = []
        pretrained_dict = dict()
        
        for key, values in model_state_dict.items():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
           
        for old_key, new_key in zip(old_keys, new_keys):
            model_state_dict[new_key] = model_state_dict.pop(old_key)
        pretrained_dict = {k: v for k, v in model_state_dict.items()}
        count = 0
        
        for key, values in model_state_dict.items():
            for count in range(args.student_hidden_layers):
                if key == "bert.encoder.layer."+str(count)+".attention.self.value.weight":
                    new_key = "bert.encoder.layer."+str(count)+".attention.self.v_2.weight"
                    pretrained_dict.update({new_key: model_state_dict[key]})
            
                if key == "bert.encoder.layer."+str(count)+".attention.self.value.bias":
                    new_key = "bert.encoder.layer."+str(count)+".attention.self.v_2.bias"
                    pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.dense.weight":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.dense.weight"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.dense.bias":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.dense.bias"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.LayerNorm.weight":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.LayerNorm.weight"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
            
#                 if key == "bert.encoder.layer."+str(count)+".output.LayerNorm.bias":
#                     new_key = "bert.encoder.layer."+str(count)+".output_2.LayerNorm.bias"
#                     pretrained_dict.update({new_key: model_state_dict[key]})
                
            #if key == "bert.encoder.layer."+str(count)+".attention.self.value.weight":
            #    neww_key = "bert.encoder.layer."+str(count)+".attention.self.v_2.weight"
            #    neww_values = values
            #    model_state_dict.update({'neww_key' : neww_values})
        
        del_keys = []
        keep_keys = []
        if mode == 'exact':
            pass
        elif mode == 'encoder':
            for t in list(pretrained_dict.keys()):
                if 'classifier' in t or 'cls' in t:
                    del pretrained_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'classifier':
            for t in list(pretrained_dict.keys()):
                if 'classifier' not in t:
                    del pretrained_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'student':
            model_keys = model.state_dict().keys()
            for t in list(pretrained_dict.keys()):
                if t not in model_keys:
                    del pretrained_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        else:
            raise ValueError('%s not available for now' % mode)

        model.load_state_dict(pretrained_dict)
        if mode != 'exact':
            logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))
        if DEBUG:
            print('deleted keys =\n {}'.format('\n'.join(del_keys)))
            print('*' * 77)
            print('kept keys =\n {}'.format('\n'.join(keep_keys)))

    if args.fp16:
        logger.info('fp16 activated, now call model.half()')
        model.half()
    model.to(device)

    if train_mode != 'finetune':
        if verbose:
            logger.info('freeze BERT layer in DEBUG mode')
        model.set_mode(train_mode)

    if local_rank != -1:
        raise NotImplementedError('not implemented for local_rank != 1')
    elif n_gpu > 1:
        logger.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model

def load_model_org(model, checkpoint, args, mode='exact', train_mode='finetune', verbose=True, DEBUG=False):
    """
    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """

    n_gpu = args.n_gpu
    device = args.device
    local_rank = -1
    if checkpoint in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint))
        model_state_dict = torch.load(checkpoint)
        old_keys = []
        new_keys = []
        for key in model_state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            model_state_dict[new_key] = model_state_dict.pop(old_key)

        del_keys = []
        keep_keys = []
        if mode == 'exact':
            pass
        elif mode == 'encoder':
            for t in list(model_state_dict.keys()):
                if 'classifier' in t or 'cls' in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'classifier':
            for t in list(model_state_dict.keys()):
                if 'classifier' not in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'student':
            model_keys = model.state_dict().keys()
            for t in list(model_state_dict.keys()):
                if t not in model_keys:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        else:
            raise ValueError('%s not available for now' % mode)
        model.load_state_dict(model_state_dict)
        if mode != 'exact':
            logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))
        if DEBUG:
            print('deleted keys =\n {}'.format('\n'.join(del_keys)))
            print('*' * 77)
            print('kept keys =\n {}'.format('\n'.join(keep_keys)))

    if args.fp16:
        logger.info('fp16 activated, now call model.half()')
        model.half()
    model.to(device)

    if train_mode != 'finetune':
        if verbose:
            logger.info('freeze BERT layer in DEBUG mode')
        model.set_mode(train_mode)

    if local_rank != -1:
        raise NotImplementedError('not implemented for local_rank != 1')
    elif n_gpu > 1:
        logger.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model

def load_model_finetune(model, layer_initialization, checkpoint, args, mode='exact', train_mode='finetune', verbose=True, DEBUG=False):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """
    
    
    n_gpu = args.n_gpu
    device = args.device
    local_rank = -1
    if checkpoint in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint))
        model_state_dict = torch.load(checkpoint)
        old_keys = []
        new_keys = []
        for key in model_state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            model_state_dict[new_key] = model_state_dict.pop(old_key)
        
            
        model_state_dict_0 = model_state_dict.copy()
        torch.set_printoptions(precision=10)
                
        for count in range(args.student_hidden_layers):
            target_layer = int(layer_initialization[count])-1
            for key in model_state_dict_0.keys():
                new_key = None
                if 'bert.encoder.layer.'+str(count)+'.' in key:
                    new_key = key.replace(str(count), str(target_layer))
                    model_state_dict.update({key: model_state_dict_0[new_key]})
                
        error = 0
        for count in range(args.student_hidden_layers):
            target_layer_num = int(layer_initialization[count])-1
            for key in model_state_dict.keys():
                if 'bert.encoder.layer.'+str(count)+'.' in key:
                    new_key = key.replace(str(count), str(target_layer_num))
                    if torch.mean(torch.abs(model_state_dict[key])) != torch.mean(torch.abs(model_state_dict_0[new_key])):
                        error+=1
                    
        if error != 0:
            print("Error has occured")
        elif error == 0:
            for count in range(args.student_hidden_layers):
                print("Layer "+str(count+1)+" = Original checkpoint's "+str(layer_initialization[count])+"-th Layer")
                
        del_keys = []
        keep_keys = []
        if mode == 'exact':
            pass
        elif mode == 'encoder':
            for t in list(model_state_dict.keys()):
                if 'classifier' in t or 'cls' in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'classifier':
            for t in list(model_state_dict.keys()):
                if 'classifier' not in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'student':
            model_keys = model.state_dict().keys()
            for t in list(model_state_dict.keys()):
                if t not in model_keys:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        else:
            raise ValueError('%s not available for now' % mode)
        
        model.load_state_dict(model_state_dict)
        
        if mode != 'exact':
            logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))
        if DEBUG:
            print('deleted keys =\n {}'.format('\n'.join(del_keys)))
            print('*' * 77)
            print('kept keys =\n {}'.format('\n'.join(keep_keys)))

    if args.fp16:
        logger.info('fp16 activated, now call model.half()')
        model.half()
    model.to(device)

    if train_mode != 'finetune':
        if verbose:
            logger.info('freeze BERT layer in DEBUG mode')
        model.set_mode(train_mode)

    if local_rank != -1:
        raise NotImplementedError('not implemented for local_rank != 1')
    elif n_gpu > 1:
        logger.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model

def load_model_NL(model, layer_initialization, checkpoint, args, mode='exact', train_mode='finetune', verbose=True, DEBUG=False):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """
    
    
    n_gpu = args.n_gpu
    device = args.device
    local_rank = -1
    if checkpoint in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint):
            raise ValueError('checkpoint %s not exist' % checkpoint)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint))
        model_state_dict = torch.load(checkpoint)
        old_keys = []
        new_keys = []
        for key in model_state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            model_state_dict[new_key] = model_state_dict.pop(old_key)
        
            
        model_state_dict_0 = model_state_dict.copy()
        torch.set_printoptions(precision=10)
        
        for count in range(18):
            for key in model_state_dict_0.keys():
                new_key = None
                if 'bert.encoder.layer.'+str(0)+'.' in key:
                    new_key = key.replace(str(0), str(count))
                    model_state_dict.update({new_key: model_state_dict_0[key]})
        
        model_state_dict_1 = model_state_dict.copy()
        
        for count in range(18):
            target_layer = int(layer_initialization[count])-1
            for key in model_state_dict_1.keys():
                new_key = None
                if 'bert.encoder.layer.'+str(count)+'.' in key:
                    new_key = key.replace(str(count), str(target_layer))
                    model_state_dict.update({key: model_state_dict_0[new_key]})
                
        error = 0
        for count in range(args.student_hidden_layers):
            target_layer_num = int(layer_initialization[count])-1
            for key in model_state_dict.keys():
                if 'bert.encoder.layer.'+str(count)+'.' in key:
                    new_key = key.replace(str(count), str(target_layer_num))
                    if torch.mean(torch.abs(model_state_dict[key])) != torch.mean(torch.abs(model_state_dict_0[new_key])):
                        error+=1
                    
        if error != 0:
            print("Error has occured")
        elif error == 0:
            for count in range(args.student_hidden_layers):
                print("Layer "+str(count+1)+" = Original checkpoint's "+str(layer_initialization[count])+"-th Layer")
                
        del_keys = []
        keep_keys = []
        if mode == 'exact':
            pass
        elif mode == 'encoder':
            for t in list(model_state_dict.keys()):
                if 'classifier' in t or 'cls' in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'classifier':
            for t in list(model_state_dict.keys()):
                if 'classifier' not in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'student':
            model_keys = model.state_dict().keys()
            for t in list(model_state_dict.keys()):
                if t not in model_keys:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        else:
            raise ValueError('%s not available for now' % mode)
   
        model.load_state_dict(model_state_dict)

        if mode != 'exact':
            logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))
        if DEBUG:
            print('deleted keys =\n {}'.format('\n'.join(del_keys)))
            print('*' * 77)
            print('kept keys =\n {}'.format('\n'.join(keep_keys)))

    if args.fp16:
        logger.info('fp16 activated, now call model.half()')
        model.half()
    model.to(device)

    if train_mode != 'finetune':
        if verbose:
            logger.info('freeze BERT layer in DEBUG mode')
        model.set_mode(train_mode)

    if local_rank != -1:
        raise NotImplementedError('not implemented for local_rank != 1')
    elif n_gpu > 1:
        logger.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model

def load_model_from_distilbert(model, layer_initialization, checkpoint_distilbert, checkpoint_bert_base, args, mode='exact', train_mode='finetune', verbose=True, DEBUG=False):
    """

    :param model:
    :param checkpoint:
    :param argstrain:
    :param mode:  this is created because for old training the encoder and classifier are mixed together
                  also adding student mode
    :param train_mode:
    :param verbose:
    :return:
    """
    
    
    n_gpu = args.n_gpu
    device = args.device
    local_rank = -1
    if checkpoint_distilbert in [None, 'None']:
        if verbose:
            logger.info('no checkpoint provided for %s!' % model._get_name())
    else:
        if not os.path.exists(checkpoint_distilbert):
            raise ValueError('checkpoint %s not exist' % checkpoint_distilbert)
        if verbose:
            logger.info('loading %s finetuned model from %s' % (model._get_name(), checkpoint_distilbert))

        model_state_dict_distilbert = torch.load(checkpoint_distilbert)
        old_keys = []
        new_keys = []
        for key in model_state_dict_distilbert.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            model_state_dict_distilbert[new_key] = model_state_dict_distilbert.pop(old_key)
            
        model_state_dict_bert_base = torch.load(checkpoint_bert_base)
        old_keys = []
        new_keys = []
        for key in model_state_dict_bert_base.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module.'):
                new_key = key.replace('module.', '')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            model_state_dict_bert_base[new_key] = model_state_dict_bert_base.pop(old_key)        
            
        model_state_dict_distilbert_0 = model_state_dict_distilbert.copy()
        model_state_dict_bert_base_0 = model_state_dict_bert_base.copy()
        torch.set_printoptions(precision=10)
                
        # first change the names of parameters from distilbert to original bert.
        for key in model_state_dict_distilbert.keys():
            new_key = None
            if 'distilbert' in key:
                new_key = key.replace('distilbert', 'bert')
                model_state_dict_distilbert_0.update({new_key: model_state_dict_distilbert[key]})
                del model_state_dict_distilbert_0[key]
        
        model_state_dict_distilbert_temp = model_state_dict_distilbert_0.copy()

        for key in model_state_dict_distilbert_temp.keys():
            new_key = None
            if 'transformer' in key:
                new_key = key.replace('transformer', 'encoder')
                model_state_dict_distilbert_0.update({new_key: model_state_dict_distilbert_temp[key]})
                del model_state_dict_distilbert_0[key]
        
        model_state_dict_distilbert_temp = model_state_dict_distilbert_0.copy()
        
        for key in model_state_dict_distilbert_temp.keys():
            new_key = None
            if 'q_lin' in key:
                new_key = key.replace('q_lin', 'self.query')
                model_state_dict_distilbert_0.update({new_key: model_state_dict_distilbert_temp[key]})
                del model_state_dict_distilbert_0[key]
            elif 'k_lin' in key:
                new_key = key.replace('k_lin', 'self.key')
                model_state_dict_distilbert_0.update({new_key: model_state_dict_distilbert_temp[key]})
                del model_state_dict_distilbert_0[key]
            elif 'v_lin' in key:
                new_key = key.replace('v_lin', 'self.value')
                model_state_dict_distilbert_0.update({new_key: model_state_dict_distilbert_temp[key]})
                del model_state_dict_distilbert_0[key]
            elif 'attention.out_lin.' in key:
                new_key = key.replace('attention.out_lin', 'attention.output.dense')
                model_state_dict_distilbert_0.update({new_key: model_state_dict_distilbert_temp[key]})
                del model_state_dict_distilbert_0[key]
            elif 'sa_layer_norm' in key:
                new_key = key.replace('sa_layer_norm', 'attention.output.LayerNorm')
                model_state_dict_distilbert_0.update({new_key: model_state_dict_distilbert_temp[key]})
                del model_state_dict_distilbert_0[key]
            elif 'ffn.lin1' in key:
                new_key = key.replace('ffn.lin1.', 'intermediate.dense.')
                model_state_dict_distilbert_0.update({new_key: model_state_dict_distilbert_temp[key]})
                del model_state_dict_distilbert_0[key]
            elif 'ffn.lin2' in key:
                new_key = key.replace('ffn.lin2.', 'output.dense.')
                model_state_dict_distilbert_0.update({new_key: model_state_dict_distilbert_temp[key]})
                del model_state_dict_distilbert_0[key]
            elif 'output_layer_norm' in key:
                new_key = key.replace('output_layer_norm', 'output.LayerNorm')
                model_state_dict_distilbert_0.update({new_key: model_state_dict_distilbert_temp[key]})
                del model_state_dict_distilbert_0[key]
            elif 'vocab_transform' in key:
                new_key = key.replace('vocab_transform', 'bert.pooler.dense')
                model_state_dict_distilbert_0.update({new_key: model_state_dict_distilbert_temp[key]})
                del model_state_dict_distilbert_0[key]        
        
        for key in model_state_dict_bert_base.keys():
            if 'token_type' in key:
                model_state_dict_distilbert_0.update({key: model_state_dict_bert_base[key]})
                
        for count in range(args.student_hidden_layers):
            target_layer = int(layer_initialization[count])-1
            for key in model_state_dict_distilbert_0.keys():
                new_key = None
                if 'bert.encoder.layer.'+str(count)+'.' in key:
                    new_key = key.replace(str(count), str(target_layer))
                    model_state_dict_distilbert_0.update({key: model_state_dict_distilbert_0[new_key]})
                
        error = 0
        for count in range(args.student_hidden_layers):
            target_layer_num = int(layer_initialization[count])-1
            for key in model_state_dict_distilbert.keys():
                if 'bert.encoder.layer.'+str(count)+'.' in key:
                    new_key = key.replace(str(count), str(target_layer_num))
                    if torch.mean(torch.abs(model_state_dict_distilbert_0[key])) != torch.mean(torch.abs(model_state_dict_distilbert[new_key])):
                        error+=1
                    
        if error != 0:
            print("Error has occured")
        elif error == 0:
            for count in range(args.student_hidden_layers):
                print("Layer "+str(count+1)+" = Original checkpoint's "+str(layer_initialization[count])+"-th Layer")
                
        del_keys = []
        keep_keys = []
        if mode == 'exact':
            pass
        elif mode == 'encoder':
            for t in list(model_state_dict.keys()):
                if 'classifier' in t or 'cls' in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'classifier':
            for t in list(model_state_dict.keys()):
                if 'classifier' not in t:
                    del model_state_dict[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        elif mode == 'student':
            model_keys = model.state_dict().keys()
            for t in list(model_state_dict_distilbert_0.keys()):
                if t not in model_keys:
                    del model_state_dict_distilbert_0[t]
                    del_keys.append(t)
                else:
                    keep_keys.append(t)
        else:
            raise ValueError('%s not available for now' % mode)
            
        model.load_state_dict(model_state_dict_distilbert_0)

        if mode != 'exact':
            logger.info('delete %d layers, keep %d layers' % (len(del_keys), len(keep_keys)))
            print('deleted keys =\n {}'.format('\n'.join(del_keys)))
            print('*' * 77)
            print('kept keys =\n {}'.format('\n'.join(keep_keys)))            
        if DEBUG:
            print('deleted keys =\n {}'.format('\n'.join(del_keys)))
            print('*' * 77)
            print('kept keys =\n {}'.format('\n'.join(keep_keys)))

    if args.fp16:
        logger.info('fp16 activated, now call model.half()')
        model.half()
    model.to(device)

    if train_mode != 'finetune':
        if verbose:
            logger.info('freeze BERT layer in DEBUG mode')
        model.set_mode(train_mode)

    if local_rank != -1:
        raise NotImplementedError('not implemented for local_rank != 1')
    elif n_gpu > 1:
        logger.info('data parallel because more than one gpu')
        model = torch.nn.DataParallel(model)
    return model

def eval_model_dataloader(encoder_bert, classifier, dataloader, device, detailed=False,
                          criterion=nn.CrossEntropyLoss(reduction='sum'), use_pooled_output=True,
                          verbose = False):
    """
    :param encoder_bert:  either a encoder, or a encoder with classifier
    :param classifier:    if a encoder, classifier needs to be provided
    :param dataloader:
    :param device:
    :param detailed:
    :return:
    """
    if hasattr(encoder_bert, 'module'):
        encoder_bert = encoder_bert.module
    if hasattr(classifier, 'module'):
        classifier = classifier.module

    n_layer = len(encoder_bert.bert.encoder.layer)
    encoder_bert.eval()
    if classifier is not None:
        classifier.eval()

    loss = 0
    acc = 0

    # set loss function
    if detailed:
        feature_maps = [[] for _ in range(n_layer)]   # assume we only deal with bert base here
        predictions = []
        pooled_feat_maps = []

    # evaluate network
    # for idx, batch in enumerate(dataloader):
    for idx, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        if len(batch) > 4:
            input_ids, input_mask, segment_ids, label_ids, *ignore = batch
        else:
            input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            if classifier is None:
                preds = encoder_bert(input_ids, segment_ids, input_mask)
            else:
                feat = encoder_bert(input_ids, segment_ids, input_mask)
                if isinstance(feat, tuple):
                    feat, pooled_feat = feat
                    if use_pooled_output:
                        preds = classifier(pooled_feat)
                    else:
                        preds = classifier(feat)
                else:
                    feat, pooled_feat = None, feat
                    preds = classifier(pooled_feat)
        loss += criterion(preds, label_ids).sum().item()
        

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(label_ids).sum().cpu().item()

        if detailed:
            bs = input_ids.shape[0]
            need_reshape = bs != pooled_feat.shape[0]
            if classifier is None:
                raise ValueError('without classifier, feature cannot be calculated')
            if feat is None:
                pass
            else:
                for fm, f in zip(feature_maps, feat):
                    if need_reshape:
                        fm.append(f.contiguous().view(bs, -1).detach().cpu().numpy())
                    else:
                        fm.append(f.detach().cpu().numpy())
            if need_reshape:
                pooled_feat_maps.append(pooled_feat.contiguous().view(bs, -1).detach().cpu().numpy())
            else:
                pooled_feat_maps.append(pooled_feat.detach().cpu().numpy())

            predictions.append(preds.detach().cpu().numpy())
        if verbose:
            logger.info('input_ids.shape = {}, tot_loss = {}, tot_correct = {}'.format(input_ids.shape, loss, acc))

    loss /= len(dataloader.dataset) * 1.0
    acc /= len(dataloader.dataset) * 1.0
    
    if detailed:
        feat_maps = [np.concatenate(t) for t in feature_maps] if len(feature_maps[0]) > 0 else None
        if n_layer == 24:
            return {'loss': loss,
                    'acc': acc,
                    'pooled_feature_maps': np.concatenate(pooled_feat_maps),
                    'pred_logit': np.concatenate(predictions),
                    'feature_maps': [feat_maps[i] for i in [3, 7, 11, 15, 19]]}
        else:
            return {'loss': loss,
                    'acc': acc,
                    'pooled_feature_maps': np.concatenate(pooled_feat_maps),
                    'pred_logit': np.concatenate(predictions),
                    'feature_maps': feat_maps}

    return {'loss': loss, 'acc': acc}

def run_process(proc):
    os.system(proc)


def eval_model_dataloader_nli(task_name, eval_label_ids, encoder_bert, classifier, dataloader, kd_model, num_labels,
                              device, weights=None, layer_idx=None, output_mode='classification'):
    encoder_bert.eval()
    classifier.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            full_output, pooled_output = encoder_bert(input_ids, segment_ids, input_mask)
            if kd_model.lower() in['kd', 'kd.cls']:
                logits = classifier(pooled_output)
            elif kd_model.lower() == 'kd.full':
                logits = classifier(full_output, weights, layer_idx)
            else:
                raise NotImplementedError(f'{kd_model} not implemented yet')

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            raise NotImplementedError('regression not implemented yet')

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1).flatten()
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    return result

def eval_model_dataloader_nli_finetune(task_name, eval_label_ids, encoder_bert, classifier, dataloader, kd_model, num_labels,
                              device, weights=None, layer_idx=None, output_mode='classification'):
    encoder_bert.eval()
    classifier.eval()

    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            full_output, pooled_output = encoder_bert(input_ids, segment_ids, input_mask)
            if kd_model.lower() in['kd', 'kd.cls']:
                logits = classifier(pooled_output)
                
            elif kd_model.lower() == 'kd.full':
                logits = classifier(full_output, weights, layer_idx)
            else:
                raise NotImplementedError(f'{kd_model} not implemented yet')

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            #loss_fct = CrossEntropyLoss()
            loss_fct=nn.CrossEntropyLoss(reduction='sum')
            #tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            
                                
        elif output_mode == "regression":
            raise NotImplementedError('regression not implemented yet')
        
        #eval_loss += tmp_eval_loss.mean().item()
        eval_loss += loss_fct(logits, label_ids).sum().item()
        nb_eval_steps += 1
        
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                
    #eval_loss = eval_loss / nb_eval_steps
    eval_loss /= len(dataloader.dataset) * 1.0
    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1).flatten()
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_label_ids.numpy())
                        
    if task_name == 'mrpc': 
        result['eval_loss'] = eval_loss
        result['f1'] = result['f1']
        result['acc_and_f1'] = result['acc_and_f1']
    
    elif task_name == 'cola':
        result['eval_loss'] = eval_loss       
        result['mcc'] = result['mcc']
    
    else:
        result['eval_loss'] = eval_loss
        
    return result
    
def eval_model_dataloader_nli_NL(task_name, eval_label_ids, encoder_bert, classifier, dataloader, kd_model, num_labels,
                              device, weights=None, layer_idx=None, output_mode='classification', NL_mode= None):
    encoder_bert.eval()
    classifier.eval()

    eval_loss = 0
    eval_loss_2 = 0
    eval_loss_3 = 0
    nb_eval_steps = 0
    nb_eval_steps_2 = 0
    nb_eval_steps_3 = 0
    preds = []
    preds_2 = []
    preds_3 = []

    for input_ids, input_mask, segment_ids, label_ids in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            full_output, pooled_output = encoder_bert(input_ids, segment_ids, input_mask, NL_mode = 0)
            full_output_2, pooled_output_2 = encoder_bert(input_ids, segment_ids, input_mask, NL_mode = 1)
            full_output_3, pooled_output_3 = encoder_bert(input_ids, segment_ids, input_mask, NL_mode = 2)
            
            if kd_model.lower() in['kd', 'kd.cls']:
                logits = classifier(pooled_output)
                logits_2 = classifier(pooled_output_2)
                logits_3 = classifier(pooled_output_3)
                
            elif kd_model.lower() == 'kd.full':
                logits = classifier(full_output, weights, layer_idx)
            else:
                raise NotImplementedError(f'{kd_model} not implemented yet')

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            #loss_fct = CrossEntropyLoss()
            loss_fct=nn.CrossEntropyLoss(reduction='sum')
            #tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            
                                
        elif output_mode == "regression":
            raise NotImplementedError('regression not implemented yet')
        
        #eval_loss += tmp_eval_loss.mean().item()
        eval_loss += loss_fct(logits, label_ids).sum().item()
        eval_loss_2 += loss_fct(logits_2, label_ids).sum().item()
        eval_loss_3 += loss_fct(logits_3, label_ids).sum().item()
        
        nb_eval_steps += 1
        nb_eval_steps_2 += 1
        nb_eval_steps_3 += 1
        
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
        if len(preds_2) == 0:
            preds_2.append(logits_2.detach().cpu().numpy())
        else:
            preds_2[0] = np.append(preds_2[0], logits_2.detach().cpu().numpy(), axis=0)            
        if len(preds_3) == 0:
            preds_3.append(logits_3.detach().cpu().numpy())
        else:
            preds_3[0] = np.append(preds_3[0], logits_3.detach().cpu().numpy(), axis=0)            
                
    #eval_loss = eval_loss / nb_eval_steps
    eval_loss /= len(dataloader.dataset) * 1.0
    eval_loss_2 /= len(dataloader.dataset) * 1.0
    eval_loss_3 /= len(dataloader.dataset) * 1.0
    preds = preds[0]
    preds_2 = preds_2[0]
    preds_3 = preds_3[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1).flatten()
        preds_2 = np.argmax(preds_2, axis=1).flatten()
        preds_3 = np.argmax(preds_3, axis=1).flatten()
    elif output_mode == "regression":
        preds = np.squeeze(preds)
        preds_2 = np.squeeze(preds_2)
        preds_3 = np.squeeze(preds_3)
        
    result = compute_metrics(task_name, preds, eval_label_ids.numpy())
    result_2 = compute_metrics(task_name, preds_2, eval_label_ids.numpy())
    result_3 = compute_metrics(task_name, preds_3, eval_label_ids.numpy())
                        
    if task_name == 'mrpc': 
        result['eval_loss_M1'] = eval_loss
        result['eval_loss_M2'] = eval_loss_2
        result['eval_loss_Neg'] = eval_loss_3
        
        result['f1_M1'] = result['f1']
        result['f1_M2'] = result_2['f1']
        result['f1_Neg'] = result_3['f1']
        
        result['acc_and_f1_M1'] = result['acc_and_f1']
        result['acc_and_f1_M2'] = result_2['acc_and_f1']
        result['acc_and_f1_Neg'] = result_3['acc_and_f1']
    
    elif task_name == 'cola':
        result['eval_loss_M1'] = eval_loss
        result['eval_loss_M2'] = eval_loss_2  
        result['eval_loss_Neg'] = eval_loss_3
        
        result['mcc_M1'] = result['mcc']
        result['mcc_M2'] = result_2['mcc']
        result['mcc_Neg'] = result_3['mcc']
    
    else:
        result['eval_loss_M1'] = eval_loss
        result['eval_loss_M2'] = eval_loss_2  
        result['eval_loss_Neg'] = eval_loss_3
        
        result['acc_M1'] = result['acc']
        result['acc_M2'] = result_2['acc']
        result['acc_Neg'] = result_3['acc']        
    return result

def eval_model_dataloader_nli_DL(task_name, eval_label_ids, encoder_bert, classifier, dataloader, kd_model, num_labels,
                              device, weights=None, layer_idx=None, output_mode='classification', NL_mode= None):
    encoder_bert.eval()
    classifier.eval()

    eval_loss = 0
    eval_loss_2 = 0
    nb_eval_steps = 0
    nb_eval_steps_2 = 0
    preds = []
    preds_2 = []

    for input_ids, input_mask, segment_ids, label_ids in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            full_output, pooled_output = encoder_bert(input_ids, segment_ids, input_mask, NL_mode = 0)
            full_output_2, pooled_output_2 = encoder_bert(input_ids, segment_ids, input_mask, NL_mode = 1)
            
            if kd_model.lower() in['kd', 'kd.cls']:
                logits = classifier(pooled_output)
                logits_2 = classifier(pooled_output_2)
                
            elif kd_model.lower() == 'kd.full':
                logits = classifier(full_output, weights, layer_idx)
            else:
                raise NotImplementedError(f'{kd_model} not implemented yet')

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            #loss_fct = CrossEntropyLoss()
            loss_fct=nn.CrossEntropyLoss(reduction='sum')
            #tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            
                                
        elif output_mode == "regression":
            raise NotImplementedError('regression not implemented yet')
        
        #eval_loss += tmp_eval_loss.mean().item()
        eval_loss += loss_fct(logits, label_ids).sum().item()
        eval_loss_2 += loss_fct(logits_2, label_ids).sum().item()
        
        nb_eval_steps += 1
        nb_eval_steps_2 += 1
        
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
        if len(preds_2) == 0:
            preds_2.append(logits_2.detach().cpu().numpy())
        else:
            preds_2[0] = np.append(preds_2[0], logits_2.detach().cpu().numpy(), axis=0)            
                
    #eval_loss = eval_loss / nb_eval_steps
    eval_loss /= len(dataloader.dataset) * 1.0
    eval_loss_2 /= len(dataloader.dataset) * 1.0
    preds = preds[0]
    preds_2 = preds_2[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1).flatten()
        preds_2 = np.argmax(preds_2, axis=1).flatten()
    elif output_mode == "regression":
        preds = np.squeeze(preds)
        preds_2 = np.squeeze(preds_2)
        
    result = compute_metrics(task_name, preds, eval_label_ids.numpy())
    result_2 = compute_metrics(task_name, preds_2, eval_label_ids.numpy())
                        
    if task_name == 'mrpc': 
        result['eval_loss_M1'] = eval_loss
        result['eval_loss_M2'] = eval_loss_2
        result['f1_M1'] = result['f1']
        result['f1_M2'] = result_2['f1']
        result['acc_and_f1_M1'] = result['acc_and_f1']
        result['acc_and_f1_M2'] = result_2['acc_and_f1']
    
    elif task_name == 'cola':
        result['eval_loss_M1'] = eval_loss       
        result['mcc_M1'] = result['mcc']
        result['eval_loss_M2'] = eval_loss_2       
        result['mcc_M2'] = result_2['mcc']
    
    else:
        result['eval_loss_M1'] = eval_loss       
        result['acc_M1'] = result['acc']
        result['eval_loss_M2'] = eval_loss_2       
        result['acc_M2'] = result_2['acc']
        
    return result