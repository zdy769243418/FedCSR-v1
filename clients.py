import numpy as np
from collections import namedtuple
import math
import gc
import logging
import numpy as np
from data_utils import load_batch, load_train_batch
import random
import os
import torch
import torch.nn as nn
import random
import os
import torch
import torch.nn as nn
import numpy as np

# from Model import AlexNet
# from Dataset import Dataset

# # The definition of fed model
# FedModel = namedtuple('FedModel', 'X Y DROP_RATE train_op loss_op acc_op')

class Clients:
    def __init__(self, model_fn, opt, adj, adj_single, train_dataloaders, test_dataloaders, valid_dataloaders):
        self.trainer = model_fn(opt, adj, adj_single)
        self.model = self.trainer.model
        self.train_dataloaders = train_dataloaders
        self.test_dataloaders = test_dataloaders
        self.valid_dataloaders = valid_dataloaders
        self.n_clients = opt["n_clients"]
        
        # 初始化各client的权重
        client_n_samples_train = [train_dataloader.num_examples  for train_dataloader in self.train_dataloaders]
        samples_sum_train = sum(client_n_samples_train)
        self.client_train_weights = [train_dataloader.num_examples /samples_sum_train for train_dataloader in self.train_dataloaders]
        # print(self.client_train_weights)
        client_n_samples_valid = [valid_dataloader.num_examples  for valid_dataloader in self.valid_dataloaders]
        samples_sum_valid = sum(client_n_samples_valid)
        self.client_valid_weights = [valid_dataloader.num_examples /samples_sum_valid for valid_dataloader in self.valid_dataloaders]
        # print(self.client_valid_weights)    

        client_n_samples_test = [test_dataloader.num_examples  for test_dataloader in self.test_dataloaders]
        samples_sum_test = sum(client_n_samples_test)
        self.client_test_weights = [test_dataloader.num_examples /samples_sum_test for test_dataloader in self.test_dataloaders]
        # print(self.client_test_weights)    
        # exit(0)


    def train_epoch(self, c_id, Round, args, source_item_num, target_item_num):
        """
            Train one client with its own data for one epoch
            cid: Client id
        """
        batchs_data = []
        for k, (batch_data) in enumerate(load_train_batch(self.train_dataloaders[c_id], args.batch_size, source_item_num, target_item_num)):
            batchs_data.append(batch_data)
        
        x_feas = []
        y_feas = []
        
        for epoch in range(args.local_epoch):
            loss = 0
            step = 0
            # print("***************")
            # print(c_id)
            # print("***************")
            i = 0
            for batch_data in batchs_data:    
                # 遍历一次dataloader需要45个batch
                # 一个batch是len18的元组
                # 然后 每个(256, 15)
                if args.fed_method == "FedProx":
                    l, x_fea, y_fea = self.trainer.train_batch(batch_data, args, x_feas, y_feas, epoch, i)
                    if epoch == 0:
                        x_feas.append(x_fea)
                        y_feas.append(y_fea)
                        
                else:
                    l = self.trainer.train_batch(batch_data, args)
                loss += l
                step += 1
                i += 1

            gc.collect()
        print('Epoch {}/{} - client {} -  Training Loss: {:.3f}'.format(Round, args.epochs, c_id, loss / step))
        # logging.info('Epoch {}/{} - client {} -  Training Loss: {:.3f}'.format(round, args.epochs, c_id, loss / step))
        return len(self.train_dataloaders[c_id])
    
    def evaluation(self, c_id, round, args, mod = "valid"):    
        # with self.graph.as_default():
        if mod == "valid":
            dataloader = self.valid_dataloaders[c_id]
        elif mod == "test":
            dataloader = self.test_dataloaders[c_id]

        X_pred, Y_pred = [], []
        for i, batch_data in enumerate(load_batch(dataloader, args.batch_size)):
            # batch: (11, 840, 15)
            X_predictions, Y_predictions = self.trainer.test_batch(batch_data)
            # (316, ) (524, )
            X_pred = X_pred + X_predictions
            Y_pred = Y_pred + Y_predictions

        gc.collect()

        # if mod == "valid":
        #     logging.info('Valid:')
        # else:
        #     logging.info('Test:')
    
        X_MRR, X_NDCG_5, X_NDCG_10, X_HR_1, X_HR_5, X_HR_10 = self.cal_test_score(X_pred)
        Y_MRR, Y_NDCG_5, Y_NDCG_10, Y_HR_1, Y_HR_5, Y_HR_10 = self.cal_test_score(Y_pred)


        return {"MRR-A": X_MRR, "HR-A @1": X_HR_1, "HR-A @5": X_HR_5, "HR-A @10":  X_HR_10 , \
                "NDCG-A @5":  X_NDCG_5, "NDCG-A @10": X_NDCG_10, "MRR-B": Y_MRR, "HR-B @1": Y_HR_1, "HR-B @5": Y_HR_5, "HR-B @10": Y_HR_10, \
                "NDCG-B @5": Y_NDCG_5, "NDCG-B @10": Y_NDCG_10}
         
    
    # def get_old_eval_res(self):
    #     return self.val_X_MRR, self.val_X_NDCG_10, self.val_X_HR_10, self.val_Y_MRR, self.val_Y_NDCG_10, self.val_Y_HR_10

    # def get_old_test_res(self):
    #     return self.test_X_MRR, self.test_X_NDCG_5, self.test_X_NDCG_10, self.test_X_HR_1, self.test_X_HR_5, self.test_X_HR_10, \
    #         self.test_Y_MRR, self.test_Y_NDCG_5, self.test_Y_NDCG_10, self.test_Y_HR_1, self.test_Y_HR_5, self.test_Y_HR_10
    
    @ staticmethod
    def cal_test_score(predictions):
        MRR=0.0
        HR_1 = 0.0
        HR_5 = 0.0
        HR_10 = 0.0
        NDCG_5 = 0.0
        NDCG_10 = 0.0
        valid_entity = 0.0
        # pdb.set_trace()
        # pred表示groundtruth物品在推荐列表中的rank
        for pred in predictions:
            valid_entity += 1
            MRR += 1 / pred
            if pred <= 1:
                HR_1 += 1
            if pred <= 5:
                NDCG_5 += 1 / np.log2(pred + 1)
                HR_5 += 1
            if pred <= 10:
                NDCG_10 += 1 / np.log2(pred + 1)
                HR_10 += 1
            # if valid_entity % 100 == 0:
            #     print('.', end='')
        return MRR/valid_entity, NDCG_5 / valid_entity, NDCG_10 / valid_entity, HR_1 / valid_entity, HR_5 / valid_entity, HR_10 / valid_entity


    def get_client_vars(self):
        """ Return all of the variables list """
        return self.model.state_dict()

    def set_global_vars(self, global_weights):
        """ Assign all of the variables with global vars """
        self.model.load_state_dict(global_weights)

    def choose_clients(self, ratio=1.0):
        """ randomly choose some clients """
        use_client_num = self.n_clients
        choose_num = math.ceil(use_client_num * ratio)
        return np.random.permutation(use_client_num)[:choose_num]

    # def get_clients_num(self):
    #     return self.n_clients
