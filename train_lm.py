import torch
from torch import nn
from torch import optim

import numpy as np

from transformers import BartConfig, T5Config
from transformers import BertTokenizer, BartTokenizer, T5Tokenizer
from transformers import EncoderDecoderModel, BartForConditionalGeneration, T5ForConditionalGeneration
from transformers import AdamW

import pdb
import argparse
import os
from tqdm import tqdm
import copy
import json
import logging


if torch.cuda.is_available(): DEVICE = 'cuda'
else: DEVICE = 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--eval_batchsize', type=int, default=32)
parser.add_argument('--data', type=str, default='tw_games/small_game_traces')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--eval_only', default=False, action='store_true')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--max_seq_len', type=int, default=1024)
parser.add_argument('--no_pretrain', default=False, action='store_true')
parser.add_argument('--save_path', type=str, default=None)
args = parser.parse_args()

pretrained = not args.no_pretrain
batchsize = args.batchsize
eval_batchsize = args.eval_batchsize
max_seq_len = args.max_seq_len
if not args.save_path:
    savePath = f'lang_models/{"pre" if pretrained else "nonpre"}_bart_lr{args.lr}_{args.data.split("/")[-1]}_data_per_cmd.p'
else:
    savePath = args.save_path
    model = torch.load(savePath)
print(savePath)


def load_data(dir_path, fnames, tokenizer, max_seq_len, max_data_size=10000):
    # for fp in os.listdir(dir_path):
    # [contexts, next utterance]
    full_data = {'contexts': [], 'tgts': [], 'tgt_states': [], 'init_state': []}  # goal + init state + actions
    actions_data = {'contexts': [], 'tgts': [], 'tgt_states': [], 'init_state': []}  # actions only
    init_actions_data = {'contexts': [], 'tgts': [], 'tgt_states': [], 'init_state': []}  # init state + actions
    for fp in tqdm(fnames):
        all_actions = []
        curr_action = []
        tgt_states = []
        # create all_actions (file, separated by commands, aka '>')
        with open(os.path.join(dir_path, f"{fp}.txt")) as f:
            approx_num_toks = 0
            for line in f:
                if line.strip() == "*** The End ***" or approx_num_toks > max_seq_len:
                    break
                if line.startswith(">"):
                    all_actions.append(''.join(curr_action))
                    curr_action = []
                curr_action.append(line)
            if line.startswith(">"):
                all_actions.append(''.join(curr_action))
            approx_num_toks += line.count(' ')
        # create tgt_states
        with open(os.path.join(dir_path, f"{fp}_states.txt")) as f:
            num_lines = 0
            for line in f:
                if num_lines > len(all_actions) + 1:  #+1 for initial state
                    break
                tgt_states.append(json.loads(line))
                num_lines += 1
        # create (context, next utterance, init_state, tgt_states) tuples for each dataset from all_actions
        # (all_actions[0], all_actions[1], tgt_states[0], tgt_states[1]);
        # (all_actions[0:1], all_actions[2], tgt_states[0], tgt_states[2]);
        # (all_actions[0:2], all_actions[3], tgt_states[0], tgt_states[3]);
        # ...
        for c in range(1,len(all_actions)):
            actions = ''.join(all_actions[1:c])
            full_data['contexts'].append(''.join([all_actions[0], actions]))
            full_data['tgts'].append(all_actions[c])
            full_data['init_state'].append(tgt_states[0])
            full_data['tgt_states'].append(tgt_states[c])

            actions_data['contexts'].append(''.join(actions))
            actions_data['tgts'].append(all_actions[c])
            actions_data['init_state'].append(tgt_states[0])
            actions_data['tgt_states'].append(tgt_states[c])

            goal = all_actions[0].split('\n')[0]
            init_actions_data['contexts'].append(''.join([all_actions[0].replace(goal, ""), actions]))
            init_actions_data['tgts'].append(all_actions[c])
            init_actions_data['init_state'].append(tgt_states[0])
            init_actions_data['tgt_states'].append(tgt_states[c])

            if len(full_data['contexts']) >= max_data_size:
                break
        if len(full_data['contexts']) >= max_data_size:
            break
    return full_data#, actions_data, init_actions_data


def convert_to_transformer_batches(args, data, tokenizer, batchsize):
    # TODO how to convert facts to string?
    state_keys_to_get = [key for key in data['init_state'][0] if key != 'facts']
    for i in range(0, len(data['contexts']), batchsize):
        context_tokens = tokenizer(data['contexts'][i:i+batchsize], return_tensors='pt', padding=True, truncation=True)
        tgt_tokens = tokenizer(data['tgts'][i:i+batchsize], return_tensors='pt', padding=True, truncation=True)
        init_states = {key: [data['init_state'][j][key] for j in range(i,i+batchsize)] for key in state_keys_to_get}
        tgt_states = {key: [data['tgt_states'][j][key] for j in range(i,i+batchsize)] for key in state_keys_to_get}
        init_state_tokens = {key: tokenizer(init_states[key], return_tensors='pt', padding=True, truncation=True).to(DEVICE) for key in init_states}
        tgt_state_tokens = {key: tokenizer(tgt_states[key], return_tensors='pt', padding=True, truncation=True).to(DEVICE) for key in tgt_states}
        yield context_tokens.to(DEVICE), tgt_tokens.to(DEVICE), init_state_tokens, tgt_state_tokens


def eval_model(args, model, dev_dataset, tokenizer, eval_batchsize):
    model.eval()
    with torch.no_grad():
        tot_val_loss = 0
        n_val = 0
        for j, (inputs, lang_tgts, init_state, tgt_state) in enumerate(tqdm(convert_to_transformer_batches(args, dev_dataset, tokenizer, eval_batchsize))):
            return_dict = model(
                input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                decoder_input_ids=lang_tgts['input_ids'], labels=lang_tgts['input_ids'], return_dict=True,
            )
            lang_loss, dec_output, encoder_hidden = return_dict.loss, return_dict.logits, return_dict.encoder_last_hidden_state

            tot_val_loss += lang_loss * len(inputs['input_ids'])
            n_val += len(inputs['input_ids'])

    print("n_val", n_val)
    avg_val_loss = tot_val_loss.item() / n_val
    return n_val, avg_val_loss


tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
if pretrained:
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base', dropout=args.dropout)
else:
    config = BartConfig.from_pretrained('facebook/bart-base')
    config.dropout = args.dropout
    model = BartForConditionalGeneration(config)
model.to(DEVICE)
optimizer = AdamW(list(model.parameters()), lr=args.lr)
print("Loaded model")

# TODO load data
dataset = load_data(args.data, ["walkthrough0"] + [f"randcmd{i}" for i in range(100)], tokenizer, max_seq_len, max_data_size=4000)
print("Loaded train data")
dev_dataset = load_data(args.data, [f"randcmd{i}" for i in range(100,200)], tokenizer, max_seq_len, max_data_size=500)
print("Loaded dev data")

# initial eval
print("Initial eval")
n_val, avg_val_loss = eval_model(args, model, dev_dataset, tokenizer, eval_batchsize)
print(f"INIT, avg val loss: {avg_val_loss}")
best_val_loss = avg_val_loss

if args.eval_only:
    exit()

# training loop
print("Start training")
for i in range(args.epochs):
    model.train()
    lang_train_losses = []
    for j, (inputs, lang_tgts, init_state, tgt_state) in enumerate(convert_to_transformer_batches(args, dataset, tokenizer, batchsize)):
        optimizer.zero_grad()
        return_dict = model(
            input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
            decoder_input_ids=lang_tgts['input_ids'], labels=lang_tgts['input_ids'], return_dict=True,
        )
        lang_loss, dec_output, encoder_hidden = return_dict.loss, return_dict.logits, return_dict.encoder_last_hidden_state
        # encoder_outputs = (encoder_hidden,)
        lang_train_losses.append(lang_loss.item())
        lang_loss.backward()
        optimizer.step()
        if j%100 == 0:
            print(f"epoch {i}, batch {j}, loss: {lang_loss.item()}", flush=True)
            # break

    n_val, avg_val_loss = eval_model(args, model, dev_dataset, tokenizer, eval_batchsize)
    print(f"epoch {i}, avg val loss: {avg_val_loss}")
    if avg_val_loss <= best_val_loss:
        print("NEW BEST MODEL")
        model.epoch = i
        best_val_loss = avg_val_loss
        torch.save(model,savePath)
        best_epoch = i
    elif avg_val_loss > best_val_loss:
        print("model val loss went up")

