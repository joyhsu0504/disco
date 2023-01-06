#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import argparse
import ipdb as pdb
import json
import random
import shutil
from termcolor import colored
import time
import copy
import imageio
from scipy.stats import entropy
import pickle

import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import h5py
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


import vr.utils as utils
import vr.preprocess
from vr.data import ClevrDataset, ClevrDataLoaderWithIndex
from vr.models import ModuleNet, Seq2Seq, LstmModel, CnnLstmModel, CnnLstmSaModel
from vr.models import FiLMedNet
from vr.models import FiLMedNetRecon, FiLMedNetReconContrastive, FiLMedNetReconContrastiveIntermediate, Generator, Discriminator
from vr.models import FiLMGen
from vr.models import StyleGanGenerator, StyleGanDiscriminator, MappingNetwork

parser = argparse.ArgumentParser()

# Input data
parser.add_argument('--train_question_h5', default='data/train_questions.h5')
parser.add_argument('--train_features_h5', default='data/train_features.h5')
parser.add_argument('--val_question_h5', default='data/val_questions.h5')
parser.add_argument('--val_features_h5', default='data/val_features.h5')
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--vocab_json', default='data/vocab.json')
parser.add_argument('--real_dataset_dir', default=None)

parser.add_argument('--loader_num_workers', type=int, default=1)
parser.add_argument('--use_local_copies', default=0, type=int)
parser.add_argument('--cleanup_local_copies', default=1, type=int)

parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=None, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='PG',
    choices=['FiLM', 'FiLMRecon', 'FiLMReconContrastiveIntermediate', 'PG', 'EE', 'PG+EE', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA'])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

# Start from an existing checkpoint
parser.add_argument('--verbose_logging', default=False, action='store_true')
parser.add_argument('--training_film', default=False, action='store_true')
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)
parser.add_argument('--load_timestep', default=False, action='store_true')

# Train options
parser.add_argument('--min_ans', default=False, action='store_true')
parser.add_argument('--entropy_threshold', default=None, type=float)

# Loss options
parser.add_argument('--r1_regularization_gamma', default=10., type=float)

# RNN options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)

# Module net / FiLMedNet options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)

# FiLM only options
parser.add_argument('--set_execution_engine_eval', default=0, type=int)
parser.add_argument('--program_generator_parameter_efficient', default=1, type=int)
parser.add_argument('--rnn_output_batchnorm', default=0, type=int)
parser.add_argument('--bidirectional', default=0, type=int)
parser.add_argument('--encoder_type', default='gru', type=str,
    choices=['linear', 'gru', 'lstm'])
parser.add_argument('--decoder_type', default='linear', type=str,
    choices=['linear', 'gru', 'lstm'])
parser.add_argument('--gamma_option', default='linear',
    choices=['linear', 'sigmoid', 'tanh', 'exp'])
parser.add_argument('--gamma_baseline', default=1, type=float)
parser.add_argument('--num_modules', default=4, type=int)
parser.add_argument('--module_stem_kernel_size', default=3, type=int)
parser.add_argument('--module_stem_stride', default=1, type=int)
parser.add_argument('--module_stem_padding', default=None, type=int)
parser.add_argument('--module_num_layers', default=1, type=int)    # Only mnl=1 currently implemented
parser.add_argument('--module_batchnorm_affine', default=0, type=int)    # 1 overrides other factors
parser.add_argument('--module_dropout', default=5e-2, type=float)
parser.add_argument('--module_input_proj', default=1, type=int)    # Inp conv kernel size (0 for None)
parser.add_argument('--module_kernel_size', default=3, type=int)
parser.add_argument('--condition_method', default='bn-film', type=str,
    choices=['block-input-film', 'block-output-film', 'bn-film', 'concat', 'conv-film', 'relu-film'])
parser.add_argument('--condition_pattern', default='', type=str)    # List of 0/1's (len = # FiLMs)
parser.add_argument('--use_gamma', default=1, type=int)
parser.add_argument('--use_beta', default=1, type=int)
parser.add_argument('--use_coords', default=1, type=int)    # 0: none, 1: low usage, 2: high usage
parser.add_argument('--grad_clip', default=0, type=float)    # <= 0 for no grad clipping
parser.add_argument('--debug_every', default=float('inf'), type=float)    # inf for no pdb
parser.add_argument('--print_verbose_every', default=float('inf'), type=float)    # inf for min print

# CNN options (for baselines)
parser.add_argument('--cnn_res_block_dim', default=128, type=int)
parser.add_argument('--cnn_num_res_blocks', default=0, type=int)
parser.add_argument('--cnn_proj_dim', default=512, type=int)
parser.add_argument('--cnn_pooling', default='maxpool2',
    choices=['none', 'maxpool2'])

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument('--classifier_downsample', default='maxpool2',
    choices=['maxpool2', 'maxpool3', 'maxpool4', 'maxpool5', 'maxpool7', 'maxpoolfull', 'none',
                     'avgpool2', 'avgpool3', 'avgpool4', 'avgpool5', 'avgpool7', 'avgpoolfull', 'aggressive'])
parser.add_argument('--classifier_fc_dims', default='1024')
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default=0, type=float)

# Optimization options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--optimizer', default='Adam',
    choices=['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'ASGD', 'RMSprop', 'SGD'])
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0, type=float)

# Output options
parser.add_argument('--checkpoint_path', default='data/checkpoint.pt')
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--avoid_checkpoint_override', default=0, type=int)
parser.add_argument('--record_loss_every', default=1, type=int)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--time', default=0, type=int)

class RealDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
            self.dataset = dataset

    def __getitem__(self, i):
            image = self.dataset[i][0]
            filename = self.dataset.samples[i][0]
            file_index = int(filename.split('/')[-1].split('.')[0].split('_')[-1])
            return image, filename

    def __len__(self):
            return len(self.dataset)


def main(args):
    print('Will save checkpoints to %s' % args.checkpoint_path)
    print('Entropy threshold ' + str(args.entropy_threshold))

    vocab = utils.load_vocab(args.vocab_json)
    if args.use_local_copies == 1:
        shutil.copy(args.train_question_h5, '/tmp/train_questions.h5')
        shutil.copy(args.train_features_h5, '/tmp/train_features.h5')
        shutil.copy(args.val_question_h5, '/tmp/val_questions.h5')
        shutil.copy(args.val_features_h5, '/tmp/val_features.h5')
        args.train_question_h5 = '/tmp/train_questions.h5'
        args.train_features_h5 = '/tmp/train_features.h5'
        args.val_question_h5 = '/tmp/val_questions.h5'
        args.val_features_h5 = '/tmp/val_features.h5'

    question_families = None
    if args.family_split_file is not None:
        with open(args.family_split_file, 'r') as f:
            question_families = json.load(f)

    train_loader_kwargs = {
        'question_h5': args.train_question_h5,
        'feature_h5': args.train_features_h5,
        'vocab': vocab,
        'batch_size': args.batch_size,
        'shuffle': True,
        'question_families': question_families,
        'max_samples': args.num_train_samples,
        'num_workers': args.loader_num_workers,
    }
    val_loader_kwargs = {
        'question_h5': args.val_question_h5,
        'feature_h5': args.val_features_h5,
        'vocab': vocab,
        'batch_size': args.batch_size,
        'question_families': question_families,
        'max_samples': args.num_val_samples,
        'num_workers': args.loader_num_workers,
    }

    with ClevrDataLoaderWithIndex(**train_loader_kwargs) as train_loader, \
             ClevrDataLoaderWithIndex(**val_loader_kwargs) as val_loader:
        train_loop(args, train_loader, val_loader)

    if args.use_local_copies == 1 and args.cleanup_local_copies == 1:
        os.remove('/tmp/train_questions.h5')
        os.remove('/tmp/train_features.h5')
        os.remove('/tmp/val_questions.h5')
        os.remove('/tmp/val_features.h5')


def train_loop(args, train_loader, val_loader): 
    vocab = utils.load_vocab(args.vocab_json)
    program_generator, pg_kwargs, pg_optimizer = None, None, None
    execution_engine, ee_kwargs, ee_optimizer = None, None, None
    baseline_model, baseline_kwargs, baseline_optimizer, baseline_type = None, None, None, None
    pg_best_state, ee_best_state, baseline_best_state = None, None, None

    # Set up model
    optim_method = getattr(torch.optim, args.optimizer)
    program_generator, pg_kwargs = get_program_generator(args)
    pg_optimizer = optim_method(program_generator.parameters(),
                                                            lr=args.learning_rate,
                                                            weight_decay=args.weight_decay)
    print('Here is the conditioning network:')
    print(program_generator)
    
    execution_engine, ee_kwargs = get_execution_engine(args)
    ee_optimizer = optim_method(execution_engine.parameters(),
                                                            lr=args.learning_rate,
                                                            weight_decay=args.weight_decay)

    print('Here is the conditioned network:')
    print(execution_engine)
    loss_fn = torch.nn.CrossEntropyLoss().cuda()    

    stats = {
        'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
        'train_accs': [], 'val_accs': [], 'val_accs_ts': [],
        'best_val_acc': -1, 'model_t': 0,
    }
    t, epoch, reward_moving_average = 0, 0, 0

    if args.execution_engine_start_from is not None and args.load_timestep:
        saved_t, saved_epoch = utils.load_t_and_epoch(args.execution_engine_start_from)
        t = saved_t
        epoch = saved_epoch
    
    writer = SummaryWriter(log_dir='path_to_metrics_logs/' + args.checkpoint_path.split('/')[-1].split('.')[0] + '/')
    set_mode('train', [program_generator, execution_engine, baseline_model])
    print('train_loader has %d samples' % len(train_loader.dataset))
    print('val_loader has %d samples' % len(val_loader.dataset))
    real_dataset = dset.ImageFolder(root=args.real_dataset_dir,
                                                         transform=transforms.Compose([
                                                                 transforms.Resize((256, 256)),
                                                                 transforms.ToTensor(),
                                                         ]))
        
    real_dataset = RealDataset(real_dataset)
    real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    print('real_loader has %d samples' % len(real_dataloader.dataset))
    real_data_generator = iter(real_dataloader)
        
    num_checkpoints = 0
    epoch_start_time = 0.0
    epoch_total_time = 0.0
    train_pass_total_time = 0.0
    val_pass_total_time = 0.0
    running_loss = 0.0
    running_dis_loss = 0.0
    running_gen_loss = 0.0
    running_disco_loss = 0.0
    total_counted_all = []
            
    with open(args.vocab_json, 'r') as f:
        vocab_file = json.load(f)
        vocab = vocab_file['question_token_to_idx']
        inverse_vocab = dict((v, k) for k, v in vocab.items())
        answer_vocab = vocab_file['answer_token_to_idx']
        inverse_answer_vocab = dict((v, k) for k, v in answer_vocab.items())
        
    scenes = 'path_to_CLEVR_scenes.json'
    print(scenes)
    with open(scenes, 'r') as f:
        file = json.load(f)
        ans = file['scenes']
        
    model = build_model()
                
    ############################
    # BEGIN UPDATES
    ###########################
    while t < args.num_iterations:
        if (epoch > 0) and (args.time == 1):
            epoch_time = time.time() - epoch_start_time
            epoch_total_time += epoch_time
            print(colored('EPOCH PASS AVG TIME: ' + str(epoch_total_time / epoch), 'white'))
            print(colored('Epoch Pass Time            : ' + str(epoch_time), 'white'))
        epoch_start_time = time.time()

        epoch += 1
        print('Starting epoch %d' % epoch)
        for batch in train_loader:
            t += 1
            questions, _, feats, answers, programs, _, _ = batch
            if isinstance(questions, list):
                questions = questions[0]
            questions_var = Variable(questions.cuda())
            feats_var = Variable(feats.cuda())
            answers_var = Variable(answers.cuda())
            if programs[0] is not None:
                programs_var = Variable(programs.cuda())
            reward = None
            if args.set_execution_engine_eval == 1:
                set_mode('eval', [execution_engine])

            ############################
            # (1) Update execution engine and program generator
            ###########################
            if args.training_film:
                if questions_var.size(0) != args.batch_size:
                    continue
                
                programs_pred = program_generator(questions_var)
                scores, z_t, z, _, _ = execution_engine(feats_var, programs_pred)

                supervised_loss = loss_fn(scores, answers_var)

                if args.training_film:
                    pg_optimizer.zero_grad()
                    ee_optimizer.zero_grad()
                    if args.debug_every <= -2:
                        pdb.set_trace()
                    supervised_loss.backward(retain_graph=True)
                    if args.debug_every < float('inf'):
                        check_grad_num_nans(execution_engine, 'FiLMedNetReconContrastiveIntermediate')
                        check_grad_num_nans(program_generator, 'FiLMGen')

                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm(program_generator.parameters(), args.grad_clip)
                    pg_optimizer.step()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm(execution_engine.parameters(), args.grad_clip)
                    ee_optimizer.step()

            ############################
            # (3) Update network: pseudo-label
            ###########################
            if not args.training_film:                   
                if questions_var.size(0) != args.batch_size:
                    continue
                     
                '''ENTROPY'''
                try:
                    real_batch, filename = next(real_data_generator)
                    real_batch = real_batch.to('cuda')
                except StopIteration:
                    real_data_generator = iter(real_dataloader)
                    real_batch, filename = next(real_data_generator)
                    real_batch = real_batch.to('cuda')    
                real_batch_extracted = real_batch * 255.0
                real_batch_extracted = transforms.Resize((224, 224))(real_batch_extracted)
                recon_feats = run_batch(real_batch_extracted, model)
                
                programs_pred = program_generator(questions_var)
                recon_scores, _, _, _, _ = execution_engine(recon_feats, programs_pred)
                
                p = torch.nn.Softmax()(recon_scores)
                entropies = -((p * p.log()).sum(dim=-1))
                entropies = entropies.cpu().detach().numpy()

                total_counted = 0
                if args.verbose_logging:
                    rofs, cas = 0, 0
                    image_names, curr_questions, pred_answers, true_answers, values, corrects = [], [], [], [], [], []
                    log_dict = {}

                _, max_pred_answers = recon_scores.data.max(1)

                disco_loss = 0.0
                for idx in range(args.batch_size):
                    if entropies[idx] > args.entropy_threshold:
                        continue

                    if args.verbose_logging:
                        referred_object_found, correct_answer, pred_answer, true_answer, curr_question, curr_filename = chosen_image_quality(filename[idx], questions_var[idx], max_pred_answers[idx], inverse_vocab, inverse_answer_vocab, ans)

                        image_names.append(curr_filename)
                        curr_questions.append(curr_question)
                        pred_answers.append(pred_answer)
                        true_answers.append(true_answer)
                        corrects.append(correct_answer)
                        values.append(entropies[idx])
                        rofs += referred_object_found
                        cas += correct_answer
                    
                    total_counted += 1
                    disco_loss += loss_fn(torch.unsqueeze(recon_scores[idx], dim=0), torch.unsqueeze(max_pred_answers[idx], dim=0))

                # For coverage
                writer.add_scalar("Loss/total_counted", total_counted, t)
                total_counted_all.append(total_counted)
                writer.add_scalar("Loss/disco_loss", disco_loss, t)

                if total_counted != 0:
                    if args.verbose_logging:
                        acc_referred_object_found = rofs / total_counted
                        acc_correct_answer = cas / total_counted
                        writer.add_scalar("Loss/acc_referred_object_found", acc_referred_object_found, t)
                        writer.add_scalar("Loss/acc_correct_answer", acc_correct_answer, t)

                    pg_optimizer.zero_grad()
                    ee_optimizer.zero_grad()
                    disco_loss.backward()
                    pg_optimizer.step()
                    ee_optimizer.step()

                if args.verbose_logging and t % 50 == 0:
                    log_dict['image_names'] = image_names
                    log_dict['questions'] = curr_questions
                    log_dict['pred_answers'] = pred_answers
                    log_dict['true_answers'] = true_answers
                    log_dict['corrects'] = corrects
                    log_dict['values'] = values
                    dlp = 'path_to_debug_logs/' + args.checkpoint_path.split('/')[-1].split('.')[0]
                    pickle.dump(log_dict, open(dlp + '_' + str(t) + '.p', 'wb'))
                
                
                '''FILM'''
                if total_counted != 0:
                    sampled_size = total_counted

                    programs_pred = program_generator(questions_var)
                    scores, z_t, z, _, _ = execution_engine(feats_var, programs_pred)
                    
                    supervised_loss = loss_fn(scores[:sampled_size], answers_var[:sampled_size])

                    pg_optimizer.zero_grad()
                    ee_optimizer.zero_grad()
                    if args.debug_every <= -2:
                        pdb.set_trace()
                    supervised_loss.backward()
                    if args.debug_every < float('inf'):
                        check_grad_num_nans(execution_engine, 'FiLMedNetReconContrastiveIntermediate')
                        check_grad_num_nans(program_generator, 'FiLMGen')

                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm(program_generator.parameters(), args.grad_clip)
                    pg_optimizer.step()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm(execution_engine.parameters(), args.grad_clip)
                    ee_optimizer.step()

            try:
                running_loss += supervised_loss.item()
            except:
                pass
            try:
                running_disco_loss = disco_loss.item()
            except:
                pass
            if t % args.record_loss_every == 0:
                avg_loss = running_loss / args.record_loss_every
                avg_disco_loss = running_disco_loss / args.record_loss_every
                print(t, avg_loss)
                stats['train_losses'].append(avg_loss)
                stats['train_losses_ts'].append(t)
                if reward is not None:
                    stats['train_rewards'].append(reward)
                running_loss = 0.0
                running_disco_loss = 0.0
                
                writer.add_scalar('Loss/avg_supervised_loss', avg_loss, int(t/args.record_loss_every))
                if args.model_type == 'FiLMReconContrastiveIntermediate':
                    writer.add_scalar("Loss/avg_disco_loss", avg_disco_loss, int(t/args.record_loss_every))
                writer.flush()
            if t % args.checkpoint_every == 0:
                num_checkpoints += 1
                print('Checking training accuracy ... ')
                start = time.time()
                train_acc = check_accuracy(args, program_generator, execution_engine,
                                                                     baseline_model, train_loader)
                if args.time == 1:
                    train_pass_time = (time.time() - start)
                    train_pass_total_time += train_pass_time
                    print(colored('TRAIN PASS AVG TIME: ' + str(train_pass_total_time / num_checkpoints), 'red'))
                    print(colored('Train Pass Time            : ' + str(train_pass_time), 'red'))
                print('train accuracy is', train_acc)
                print('Checking validation accuracy ...')
                start = time.time()
                val_acc = check_accuracy(args, program_generator, execution_engine,
                                                                 baseline_model, val_loader)
                if args.time == 1:
                    val_pass_time = (time.time() - start)
                    val_pass_total_time += val_pass_time
                    print(colored('VAL PASS AVG TIME:     ' + str(val_pass_total_time / num_checkpoints), 'cyan'))
                    print(colored('Val Pass Time                : ' + str(val_pass_time), 'cyan'))
                print('val accuracy is ', val_acc)
                stats['train_accs'].append(train_acc)
                stats['val_accs'].append(val_acc)
                stats['val_accs_ts'].append(t)
                
                writer.add_scalar("Acc/train_acc", train_acc, int(t/args.record_loss_every))
                writer.add_scalar("Acc/val_acc", val_acc, int(t/args.record_loss_every))

                if val_acc > stats['best_val_acc']:
                    stats['best_val_acc'] = val_acc
                    stats['model_t'] = t
                    
                total_counted_all_avg = float(sum(total_counted_all)) / len(total_counted_all)
                print('total counted in batch average is ' + str(total_counted_all_avg))
                total_counted_all = []

                curr_pg_state = get_state(program_generator)
                curr_ee_state = get_state(execution_engine)
                curr_baseline_state = get_state(baseline_model)
                curr_gen_state = None
                curr_dis_state = None

                checkpoint = {
                    'args': args.__dict__,
                    'program_generator_kwargs': pg_kwargs,
                    'program_generator_state': curr_pg_state,
                    'execution_engine_kwargs': ee_kwargs,
                    'execution_engine_state': curr_ee_state,
                    'generator_state': curr_gen_state,
                    'discriminator_state': curr_dis_state,
                    'baseline_kwargs': baseline_kwargs,
                    'baseline_state': curr_baseline_state,
                    'baseline_type': baseline_type,
                    'vocab': vocab,
                    't': t,
                    'epoch': epoch
                }
                for k, v in stats.items():
                    checkpoint[k] = v
                checkpoint_timestep_path = args.checkpoint_path.split('.')[0] + '_' + str(int(t / args.checkpoint_every)) + '.pt'
                print('Saving checkpoint to %s' % checkpoint_timestep_path)
                torch.save(checkpoint, checkpoint_timestep_path)
                del checkpoint['program_generator_state']
                del checkpoint['execution_engine_state']
                del checkpoint['generator_state']
                del checkpoint['discriminator_state']
                del checkpoint['baseline_state']
                with open(args.checkpoint_path + '.json', 'w') as f:
                    json.dump(checkpoint, f)

            if t == args.num_iterations:
                break
    writer.close()

    
def chosen_image_quality(fn, q, answer, inverse_vocab, inverse_answer_vocab, ans):
    q = q.cpu().detach().numpy()
    answer = answer.cpu().detach().numpy()
        
    attr = inverse_vocab[q[2]]
    referred = inverse_vocab[q[5]]
    
    filename = fn.split('/')[-1]
    pred_answer = inverse_answer_vocab[int(answer)]
    true_answer = 'DNE'
            
    for a in ans:
        if a['image_filename'] == filename:
            objects = a['objects']
            referred_object_found = False
            correct_answer = False
            for o in objects:
                if o['shape'] == referred or o['size'] == referred or o['color'] == referred or o['material'] == referred:
                    referred_object_found = True
                    true_answer = o[attr]

                    if pred_answer == true_answer:
                        correct_answer = True

            break
            
    if referred_object_found:
        rof = 1
    else:
        rof = 0
        
    if correct_answer:
        ca = 1
    else:
        ca = 0
    return rof, ca, pred_answer, true_answer, q, filename
    

def nonsat_loss_fn(inputs, label=None):
    # Non-saturating loss
    l = -1 if label else 1
    return F.softplus(l*inputs).mean()


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    

def build_model():
    model = 'resnet101'
    model_stage = 3
    if not hasattr(torchvision.models, model):
        raise ValueError('Invalid model "%s"' % model)
    cnn = getattr(torchvision.models, model)(pretrained=True)
    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]
    for i in range(model_stage):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)
    for param in model.parameters():
        param.requires_grad = False
    model.cuda()
    model.eval()
    return model

                
def run_batch(image_batch, model):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
    mean = torch.from_numpy(mean).to(device='cuda', dtype=torch.float)
    std = torch.from_numpy(std).to(device='cuda', dtype=torch.float)
    max_float = torch.FloatTensor([255.0]).cuda()

    image_batch = (image_batch / max_float - mean) / std

    feats = model(image_batch)

    return feats
                
                
def parse_int_list(s):
    if s == '': return ()
    return tuple(int(n) for n in s.split(','))


def get_state(m):
    if m is None:
        return None
    state = {}
    for k, v in m.state_dict().items():
        state[k] = v.clone()
    return state


def get_program_generator(args):
    vocab = utils.load_vocab(args.vocab_json)
    if args.program_generator_start_from is not None:
        pg, kwargs = utils.load_program_generator(
            args.program_generator_start_from, model_type=args.model_type)
        cur_vocab_size = pg.encoder_embed.weight.size(0)
        if cur_vocab_size != len(vocab['question_token_to_idx']):
            print('Expanding vocabulary of program generator')
            pg.expand_encoder_vocab(vocab['question_token_to_idx'])
            kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
    else:
        kwargs = {
            'encoder_vocab_size': len(vocab['question_token_to_idx']),
            'decoder_vocab_size': len(vocab['program_token_to_idx']),
            'wordvec_dim': args.rnn_wordvec_dim,
            'hidden_dim': args.rnn_hidden_dim,
            'rnn_num_layers': args.rnn_num_layers,
            'rnn_dropout': args.rnn_dropout,
        }
        if args.model_type in ['FiLM', 'FiLMRecon', 'FiLMReconContrastiveIntermediate']:
            kwargs['parameter_efficient'] = args.program_generator_parameter_efficient == 1
            kwargs['output_batchnorm'] = args.rnn_output_batchnorm == 1
            kwargs['bidirectional'] = args.bidirectional == 1
            kwargs['encoder_type'] = args.encoder_type
            kwargs['decoder_type'] = args.decoder_type
            kwargs['gamma_option'] = args.gamma_option
            kwargs['gamma_baseline'] = args.gamma_baseline
            kwargs['num_modules'] = args.num_modules
            kwargs['module_num_layers'] = args.module_num_layers
            kwargs['module_dim'] = args.module_dim
            kwargs['debug_every'] = args.debug_every
            pg = FiLMGen(**kwargs)
    pg.cuda()
    pg.train()
    return pg, kwargs


def get_execution_engine(args):
    vocab = utils.load_vocab(args.vocab_json)
    if args.execution_engine_start_from is not None:
        ee, kwargs = utils.load_execution_engine(
            args.execution_engine_start_from, model_type=args.model_type)
        ee.cuda()
        ee.train()
        return ee, kwargs
    
    else:
        kwargs = {
            'vocab': vocab,
            'feature_dim': parse_int_list(args.feature_dim),
            'stem_batchnorm': args.module_stem_batchnorm == 1,
            'stem_num_layers': args.module_stem_num_layers,
            'module_dim': args.module_dim,
            'module_residual': args.module_residual == 1,
            'module_batchnorm': args.module_batchnorm == 1,
            'classifier_proj_dim': args.classifier_proj_dim,
            'classifier_downsample': args.classifier_downsample,
            'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
            'classifier_batchnorm': args.classifier_batchnorm == 1,
            'classifier_dropout': args.classifier_dropout,
        }
        
        if args.model_type == 'FiLMReconContrastiveIntermediate':
            kwargs['num_modules'] = args.num_modules
            kwargs['stem_kernel_size'] = args.module_stem_kernel_size
            kwargs['stem_stride'] = args.module_stem_stride
            kwargs['stem_padding'] = args.module_stem_padding
            kwargs['module_num_layers'] = args.module_num_layers
            kwargs['module_batchnorm_affine'] = args.module_batchnorm_affine == 1
            kwargs['module_dropout'] = args.module_dropout
            kwargs['module_input_proj'] = args.module_input_proj
            kwargs['module_kernel_size'] = args.module_kernel_size
            kwargs['use_gamma'] = args.use_gamma == 1
            kwargs['use_beta'] = args.use_beta == 1
            kwargs['use_coords'] = args.use_coords
            kwargs['debug_every'] = args.debug_every
            kwargs['print_verbose_every'] = args.print_verbose_every
            kwargs['condition_method'] = args.condition_method
            kwargs['condition_pattern'] = parse_int_list(args.condition_pattern)
            ee = FiLMedNetReconContrastiveIntermediate(**kwargs)
            
            ee.cuda()
            ee.train()
            return ee, kwargs


def set_mode(mode, models):
    assert mode in ['train', 'eval']
    for m in models:
        if m is None: continue
        if mode == 'train': m.train()
        if mode == 'eval': m.eval()


def check_accuracy(args, program_generator, execution_engine, baseline_model, loader):
    set_mode('eval', [program_generator, execution_engine, baseline_model])
    num_correct, num_samples = 0, 0
    for batch in loader:
        questions, _, feats, answers, programs, _, curr_image_index = batch
        if isinstance(questions, list):
            questions = questions[0]

        questions_var = Variable(questions.cuda(), volatile=True)
        feats_var = Variable(feats.cuda(), volatile=True)
        answers_var = Variable(feats.cuda(), volatile=True)
        if programs[0] is not None:
            programs_var = Variable(programs.cuda(), volatile=True)

        scores = None
        programs_pred = program_generator(questions_var)
        scores, _, _, _, _ = execution_engine(feats_var, programs_pred)
        
        if scores is not None:
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == answers).sum()
            num_samples += preds.size(0)

        if args.num_val_samples is not None and num_samples >= args.num_val_samples:
            break

    set_mode('train', [program_generator, execution_engine, baseline_model])
    acc = float(num_correct) / num_samples
    return acc


def check_grad_num_nans(model, model_name='model'):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        num_nans = [np.sum(np.isnan(grad.data.cpu().numpy())) for grad in grads]
        nan_checks = [num_nan == 0 for num_nan in num_nans]
        if False in nan_checks:
            print('Nans in ' + model_name + ' gradient!')
            print(num_nans)
            pdb.set_trace()
            raise(Exception)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



