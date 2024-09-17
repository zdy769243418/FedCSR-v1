# -*- coding: utf-8 -*-

import sys
from datetime import datetime
import time

import argparse
from shutil import copyfile

import torch.optim as optim
from torch.autograd import Variable
from utils import torch_utils, helper
from utils.GraphMaker import GraphMaker
from model.trainer import CDSRTrainer
from utils.loader import *
import json
import codecs
from tqdm import tqdm
import pdb
import logging
from clients import Clients
from data_utils import load_dataset, preprocess_dataset
import random
import os
import torch
import torch.nn as nn
import numpy as np
from aggregator import Aggregator
from torch.optim.lr_scheduler import ExponentialLR

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


def l2_reg(params1, params2, mu):
    #params2 = [params2[key] for key in params1.keys()]
    s1 = flatten(params1) # 多维参数要用flatten压成一维的
    s2 = flatten(params2)
    return mu/2 * torch.norm(s1 - s2)

# def seed_everything(seed=1111):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
#     np.random.seed(seed)  # Numpy module.
#     random.seed(seed)  # Python random module.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     os.environ['PYTHONHASHSEED'] = str(seed)


def init_logger(args):
    log_file = os.path.join(args.log_dir, args.data_dir + '.log')

    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode='w+'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def arg_parse():
    parser = argparse.ArgumentParser()
    # dataset part
    parser.add_argument('--data_dir', type=str, default='Movie-Book', help='Movie-Book, Entertainment-Education')
    parser.add_argument('--split', type=str, default='both', help='origin, quantity, quality, both')
    
    # model part
    parser.add_argument('--model', type=str, default="C2DSR", help='model name')
    parser.add_argument('--hidden_units', type=int, default=256, help='lantent dim.')
    parser.add_argument('--num_blocks', type=int, default=2, help='lantent dim.')
    parser.add_argument('--num_heads', type=int, default=1, help='lantent dim.')
    parser.add_argument('--GNN', type=int, default=1, help='GNN depth.')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate.')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
    parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--leakey', type=float, default=0.1)
    parser.add_argument('--maxlen', type=int, default=15)
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--lambda', type=float, default=0.5)

    # train part
    parser.add_argument('--epochs', type=int, default=50, help='Number of total training iterations.')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size.')
    parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k iterations.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
    parser.add_argument('--seed', type=int, default=2040)
    parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
    parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
    parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
    parser.add_argument('--undebug', action='store_false', default=True)
    parser.add_argument('--n_clients', type=int, default=10)
    parser.add_argument('--local_epoch', type=int, default=3, help='Number of local training epochs.')
    parser.add_argument('--eval_interval', type=int, default=1, help='Interval of evalution')
    parser.add_argument('--frac', type=float, default=1, help='Fraction of participating clients')
    parser.add_argument('--log_dir', type=str, default='log', help='directory of logs')
    parser.add_argument("--preprocess", dest="preprocess", action="store_true", default=True, help="need to preprocess data") 
    parser.add_argument('--fed_method', type=str, default="FedProx", help='`FedAvg` or `FedProx`')
    parser.add_argument('--mu', type=float, default=0.01, help='hyper parameter for FedProx')
    
    args = parser.parse_args()
    return args


def evaluation_logging(eval_logs, Round, mod="valid", print_A=False, print_B=False):
#     if mod == "valid":
#         logging.info('Epoch%d Valid:' % Round)
#     else:
#         logging.info('Test:')
    avg_eval_log = {}
    avg_eval_log['Epoch'] = Round    

    for key in eval_logs[0].keys():
        avg_eval_val = 0
        for i in range(len(eval_logs)):
            avg_eval_val += eval_logs[i][key]
        avg_eval_log[key] = avg_eval_val
    
#     if print_A:
#         logging.info('MRR-A: ' + str(avg_eval_log["MRR-A"]))
#         logging.info('HR-A @1|5|10: ' + str(avg_eval_log["HR-A @1"]) + '\t' + str(avg_eval_log["HR-A @5"]) + '\t' + str(\
#             avg_eval_log["HR-A @10"]) + '\t')
#         logging.info('NDCG-A @5|10: ' + str(avg_eval_log["NDCG-A @5"]) + '\t' + str(\
#             avg_eval_log["NDCG-A @10"]) + '\t')

#     if print_B:
#         logging.info('MRR-B: ' + str(avg_eval_log["MRR-B"]))
#         logging.info('HR-B @1|5|10: ' + str(avg_eval_log["HR-B @1"]) + '\t' + str(avg_eval_log["HR-B @5"]) + '\t' + str(\
#             avg_eval_log["HR-B @10"]) + '\t')
#         logging.info('NDCG-B @5|10: ' + str(avg_eval_log["NDCG-B @5"]) + '\t' + str(\
#             avg_eval_log["NDCG-B @10"]) + '\t')
#         logging.info("")
    #print(avg_eval_log)

    return avg_eval_log


def main():
    args = arg_parse()
    
#     if args.cpu:
#         args.cuda = False
#     elif args.cuda:
#         torch.cuda.manual_seed(args.seed)

    # init_logger(args)
    # make opt
    opt = vars(args)

    model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
    model_save_dir = opt['save_dir'] + '/' + model_id
    opt['model_save_dir'] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)
    # save config
    helper.save_config(opt, model_save_dir + '/config.json', verbose=True)

    # print model info
    helper.print_config(opt)

    if opt["undebug"]:
        pass

    print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))

    if opt['preprocess']:
        preprocess_dataset(opt)

    train_dataloaders, valid_dataloaders, test_dataloaders = load_dataset(opt)


    n_clients = opt["n_clients"]

    print("Data loading done!")

    opt["itemnum"] = opt["source_item_num"] + opt["target_item_num"] + 1


    filename = opt["data_dir"]
    train_data = "./dataset/" + filename + "/traindata_new.txt"
    G = GraphMaker(opt, train_data)
    adj, adj_single = G.adj, G.adj_single
    print("graph loaded!")


    if opt["cuda"]:
        torch.cuda.empty_cache()
        adj = adj.cuda()
        adj_single = adj_single.cuda()

    if not opt['load']:
        clients = Clients(CDSRTrainer, opt, adj, adj_single, train_dataloaders, valid_dataloaders, test_dataloaders)
    else:
        exit(0)

    global_vars = clients.get_client_vars()

    best_val = 0
    rounds_without_improvement = 0
    patience = 20 #早停次数

    file = open('result.txt', 'w')
    
    for Round in range(1, opt['epochs'] + 1):
        # We are going to sum up active clients' vars at each epoch
        client_vars_sum = None
#         random_clients = clients.choose_clients(args.frac)
        random_clients = [0,1,2,3,4,5,6,7,8,9]    
        # Train with these clients
        clients_var = {}
        for c_id in tqdm(random_clients, ascii=True):
            # Restore global vars to client's model
            clients.set_global_vars(global_vars)
            # train one client
            clients.train_epoch(c_id, Round, args, opt["source_item_num"], opt["target_item_num"])
            # obtain current client's vars
            current_client_vars = copy.deepcopy(clients.get_client_vars())
            clients_var[c_id] = current_client_vars
            # sum it up with weights
            if client_vars_sum is None:
                client_vars_sum = dict((key, value * clients.client_train_weights[c_id] ) for key, value in current_client_vars.items())
            else:
                for key in client_vars_sum.keys():
                    client_vars_sum[key] += clients.client_train_weights[c_id] * current_client_vars[key]  
                     
        global_vars = client_vars_sum
        if Round % args.eval_interval == 0:
            clients.set_global_vars(global_vars)
            eval_logs = []
            random_clients = [c_id for c_id in range(args.n_clients)]

            for c_id in tqdm(random_clients, ascii=True):
                eval_log = clients.evaluation(c_id, Round, args, mod = "valid")
                eval_logs.append(dict((key, value * clients.client_valid_weights[c_id]) for key, value in eval_log.items()))

            avg_eval_log = evaluation_logging(eval_logs, Round, mod="valid", print_A=True, print_B=True)
            
            if (avg_eval_log["MRR-A"] +  avg_eval_log["MRR-B"]) > best_val:
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
                
            best_val = avg_eval_log["MRR-A"] +  avg_eval_log["MRR-B"]
            eval_logs = []
            for c_id in tqdm(random_clients, ascii=True):
                eval_log = clients.evaluation(c_id, Round, args, mod = "test")
                eval_logs.append(dict((key, value * clients.client_test_weights[c_id]) for key, value in eval_log.items()))

            result = evaluation_logging(eval_logs, Round, mod="test", print_A=True, print_B=True)
            result_str = json.dumps(result)
            file.write(result_str + '\n')
            print(result)    
                
            if rounds_without_improvement ==  patience:          
                print('Early stopping at Round {}...'.format(Round))
                torch.save(clients.model.state_dict(), "final-model_parameter.pkl")
                break          
                
    file.close()

if __name__ == "__main__":
    seed = 1111  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    main()