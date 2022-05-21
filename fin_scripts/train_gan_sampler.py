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
from scipy.stats import entropy

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
parser.add_argument('--training_gan', default=False, action='store_true')
parser.add_argument('--training_film', default=False, action='store_true')
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--stylegan_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)
parser.add_argument('--load_timestep', default=False, action='store_true')

# Train options
parser.add_argument('--freeze_gan_synthesis', default=False, action='store_true')
parser.add_argument('--entropy_threshold', default=None, type=float)

# Contrastive loss options
parser.add_argument('--contrastive_loss', default='TaskVGenContrastive', type=str,
    choices=['TaskVGenContrastive', 'QuestionVDiffQContrastive', 'JointContrastive'])
parser.add_argument('--contrastive_margin', default=0.3, type=float)
parser.add_argument('--contrastive_alpha', default=0.5, type=float)
parser.add_argument('--contrastive_beta', default=0.0001, type=float)
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
            return image

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
        'shuffle': True, # args.shuffle_train_data == 1,
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
    gen_optimizer, dis_optimizer = None, None

    # Set up model
    optim_method = getattr(torch.optim, args.optimizer)
    program_generator, pg_kwargs = get_program_generator(args)
    pg_optimizer = optim_method(program_generator.parameters(),
                                                            lr=args.learning_rate,
                                                            weight_decay=args.weight_decay)
    print('Here is the conditioning network:')
    print(program_generator)
    
    execution_engine, ee_kwargs, gen, dis = get_execution_engine(args) 
    ee_optimizer = optim_method(execution_engine.parameters(),
                                                            lr=args.learning_rate,
                                                            weight_decay=args.weight_decay)
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.002, betas=(0,0.99), eps=1e-8)
    dis_optimizer = torch.optim.Adam(dis.parameters(), lr=0.002, betas=(0,0.99), eps=1e-8)
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
        saved_t, saved_epoch = utils.load_t_and_epoch(args.stylegan_start_from) # execution_engine_start_from
        t = saved_t
        epoch = saved_epoch
    
    writer = SummaryWriter(log_dir='/viscam/u/joycj/dvr/film/metrics_logs/' + args.checkpoint_path.split('/')[-1].split('.')[0] + '/')
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
    running_contrastive_loss = 0.0
    total_counted_all = []
        
    if args.freeze_gan_synthesis:
        for param in gen.synthesis.parameters():
            param.requires_grad = False
            
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
            # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z))), Update G network: maximize log(D(G(z)))
            ###########################
            if args.training_gan:
                ## Train with all-real batch
                try:
                    real_batch = next(real_data_generator)
                    real_batch = real_batch.to('cuda')
                except StopIteration:
                    real_data_generator = iter(real_dataloader)
                    real_batch = next(real_data_generator)
                    real_batch = real_batch.to('cuda')
                real_batch.requires_grad = True
                if real_batch.size(0) != args.batch_size:
                    continue

                real_batch = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.224))(real_batch)
                output = dis(real_batch, None).view(-1)

                errD_real = nonsat_loss_fn(output, True)
                grad_real = torch.autograd.grad(outputs=output.sum(), inputs=real_batch, create_graph=True)[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = 0.5 * args.r1_regularization_gamma * grad_penalty
                errD_real += grad_penalty    

                ## Train with all-fake batch
                gaussian_noise = torch.randn(args.batch_size, 512).to('cuda')
                gen_intermediate = gen(gaussian_noise, None).detach()
                output = dis(gen_intermediate, None).view(-1)
                
                errD_fake = nonsat_loss_fn(output, False)
                errD = errD_real + errD_fake

                dis.zero_grad()
                errD_real.backward()
                errD_fake.backward()
                dis_optimizer.step()

                ## Train generator
                gen_intermediate = gen(gaussian_noise, None)
                output = dis(gen_intermediate, None).view(-1)
                
                errG = nonsat_loss_fn(output, True)

                gen.zero_grad()
                errG.backward()
                gen_optimizer.step()

            ############################
            # (3) Update network: contrastive loss
            ###########################
            if not args.training_film and not args.training_gan:    
                if questions_var.size(0) != args.batch_size:
                    continue
 
                '''ENTROPY'''
                gaussian_noise = torch.randn(args.batch_size, 512).to('cuda')
                recon_img = gen(gaussian_noise, None)

                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
                std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
                mean = torch.from_numpy(mean).to(device='cuda', dtype=torch.float)
                std = torch.from_numpy(std).to(device='cuda', dtype=torch.float)
                recon_img = (recon_img * std) + mean
                max_float = torch.FloatTensor([255.0]).cuda()
                recon_img = recon_img * max_float
                recon_img = transforms.Resize((224, 224))(recon_img)
                recon_feats = run_batch(recon_img, model)
                
                programs_pred = program_generator(questions_var)
                recon_scores, _, _, _, _ = execution_engine(recon_feats, programs_pred)
                
                p = torch.nn.Softmax()(recon_scores)
                entropies = -((p * p.log()).sum(dim=-1))
                entropies = entropies.cpu().detach().numpy()
                
                total_counted = 0
                _, max_pred_answers = recon_scores.data.max(1)
                
                contrastive_loss = 0.0
                for idx in range(args.batch_size):
                    if entropies[idx] > args.entropy_threshold:
                        continue
                        
                    total_counted += 1
                    contrastive_loss += loss_fn(torch.unsqueeze(recon_scores[idx], dim=0), torch.unsqueeze(max_pred_answers[idx], dim=0))
                
                # For coverage
                writer.add_scalar("Loss/total_counted", total_counted, t)
                total_counted_all.append(total_counted)
                writer.add_scalar("Loss/contrastive_loss", contrastive_loss, t)
                
                if total_counted != 0:
                    pg_optimizer.zero_grad()
                    ee_optimizer.zero_grad()
                    contrastive_loss.backward()
                    pg_optimizer.step()
                    ee_optimizer.step()
                    
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

            # Training logs
            if t % args.checkpoint_every == 0 and not args.training_film:
                plt.figure(figsize=(8,8))
                plt.axis('off')
                
                gen_out = gen(gaussian_noise, None).detach()

                plt.imshow(np.transpose(vutils.make_grid(gen_out.to('cuda')[:64], padding=2, normalize=True).cpu(),(1,2,0)))
                save_dir = '/viscam/u/joycj/dvr/film/recon_logs/' + args.checkpoint_path.split('/')[-1].split('.')[0] + '/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                plt.savefig(save_dir + str(int(t / args.checkpoint_every)).zfill(4) + '.png')

            try:
                running_loss += supervised_loss.item()
            except:
                pass
            try:
                running_dis_loss += errD.item()
            except:
                pass
            try:
                running_gen_loss += errG.item()
            except:
                pass
            try:
                running_contrastive_loss = contrastive_loss.item()
            except:
                pass
            if t % args.record_loss_every == 0:
                avg_loss = running_loss / args.record_loss_every
                avg_dis_loss = running_dis_loss / args.record_loss_every
                avg_gen_loss = running_gen_loss / args.record_loss_every
                avg_contrastive_loss = running_contrastive_loss / args.record_loss_every
                print(t, avg_loss)
                stats['train_losses'].append(avg_loss)
                stats['train_losses_ts'].append(t)
                if reward is not None:
                    stats['train_rewards'].append(reward)
                running_loss = 0.0
                running_dis_loss = 0.0
                running_gen_loss = 0.0
                running_contrastive_loss = 0.0
                
                writer.add_scalar('Loss/avg_supervised_loss', avg_loss, int(t/args.record_loss_every))
                writer.add_scalar("Loss/avg_dis_loss", avg_dis_loss, int(t/args.record_loss_every))
                writer.add_scalar("Loss/avg_gen_loss", avg_gen_loss, int(t/args.record_loss_every))
                if args.model_type == 'FiLMReconContrastiveIntermediate':
                    writer.add_scalar("Loss/avg_contrastive_loss", avg_contrastive_loss, int(t/args.record_loss_every))
                writer.flush()
            if t % args.checkpoint_every == 0:
                num_checkpoints += 1
                
                if not args.training_gan:
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
                curr_gen_state = get_state(gen)
                curr_dis_state = get_state(dis)

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
            
        gen = utils.load_gen_stylegan_conditional_one_hot(args.stylegan_start_from, 0)
        dis = utils.load_dis_stylegan_conditional_one_hot(args.stylegan_start_from, 0)
        
        ee.cuda()
        ee.train()
        gen.cuda()
        gen.train()
        dis.cuda()
        dis.train()
        return ee, kwargs, gen, dis
    
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
        
            c_dim = 0
            
            print('Creating StyleGAN')
            mapping_kwargs = {'num_layers': 8}
            synthesis_kwargs = {'noise_mode': 'const', 'force_fp32': False}
            gen = StyleGanGenerator(z_dim=512, c_dim=(c_dim), w_dim=512, img_resolution=256, img_channels=3, mapping_kwargs=mapping_kwargs)
                
            block_kwargs = {'freeze_layers': 0}
            mapping_kwargs = {'num_layers': 8}
            epilogue_kwargs = {'mbstd_group_size': 4}
            dis = StyleGanDiscriminator(c_dim=(c_dim), img_resolution=256,img_channels=3, architecture='resnet', channel_base=32768, channel_max=512, num_fp16_res=4, conv_clamp=256, cmap_dim=None, block_kwargs=block_kwargs, mapping_kwargs=mapping_kwargs, epilogue_kwargs=epilogue_kwargs)
        
            ee.cuda()
            ee.train()
            gen.cuda()
            gen.train()
            dis.cuda()
            dis.train()
            return ee, kwargs, gen, dis


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




