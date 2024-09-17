import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import torch_utils
from model.CDSR import C2DSR
import pdb
import numpy as np
import random
import os
import torch
import torch.nn as nn
import random
import os
import torch
import torch.nn as nn
import numpy as np


class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class CDSRTrainer(Trainer):
    def __init__(self, opt, adj = None, adj_single = None):
        self.opt = opt
        if opt["model"] == "C2DSR":
            self.model = C2DSR(opt, adj, adj_single)
            # self.cl = CL(dim=256, pred_dim=64)
        else:
            print("please select a valid model")
            exit(0)

        self.mi_loss = 0
        self.BCE_criterion = nn.BCEWithLogitsLoss()
        self.CS_criterion = nn.CrossEntropyLoss(reduction='none')
        self.criterion = nn.CosineSimilarity(dim=1) # 损失函数定义，余弦相似性
        if opt['cuda']:
            self.model.cuda()
            self.BCE_criterion.cuda()
            self.CS_criterion.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'])

    def get_dot_score(self, A_embedding, B_embedding):
        output = (A_embedding * B_embedding).sum(dim=-1)
        return output

    def unpack_batch_predict(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            X_last = inputs[6]
            Y_last = inputs[7]
            XorY = inputs[8]
            ground_truth = inputs[9]
            neg_list = inputs[10]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            X_last = inputs[6]
            Y_last = inputs[7]
            XorY = inputs[8]
            ground_truth = inputs[9]
            neg_list = inputs[10]
        return seq, x_seq, y_seq, position, x_position, y_position, X_last, Y_last, XorY, ground_truth, neg_list

    def unpack_batch(self, batch):
        if self.opt["cuda"]:
            inputs = [Variable(b.cuda()) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ground = inputs[6]
            share_x_ground = inputs[7]
            share_y_ground = inputs[8]
            x_ground = inputs[9]
            y_ground = inputs[10]
            ground_mask = inputs[11]
            share_x_ground_mask = inputs[12]
            share_y_ground_mask = inputs[13]
            x_ground_mask = inputs[14]
            y_ground_mask = inputs[15]
            corru_x = inputs[16]
            corru_y = inputs[17]
             
            seq1 = inputs[18]
            x_seq1 = inputs[19]
            y_seq1 = inputs[20]
            position1 = inputs[21]
            x_position1 = inputs[22]
            y_position1 = inputs[23]
            ground1 = inputs[24]
            share_x_ground1 = inputs[25]
            share_y_ground1 = inputs[26]
            x_ground1 = inputs[27]
            y_ground1 = inputs[28]
            ground_mask1 = inputs[29]
            share_x_ground_mask1 = inputs[30]
            share_y_ground_mask1 = inputs[31]
            x_ground_mask1 = inputs[32]
            y_ground_mask1 = inputs[33]
            corru_x1 = inputs[34]
            corru_y1 = inputs[35]
        else:
            inputs = [Variable(b) for b in batch]
            seq = inputs[0]
            x_seq = inputs[1]
            y_seq = inputs[2]
            position = inputs[3]
            x_position = inputs[4]
            y_position = inputs[5]
            ground = inputs[6]
            share_x_ground = inputs[7]
            share_y_ground = inputs[8]
            x_ground = inputs[9]
            y_ground = inputs[10]
            ground_mask = inputs[11]
            share_x_ground_mask = inputs[12]
            share_y_ground_mask = inputs[13]
            x_ground_mask = inputs[14]
            y_ground_mask = inputs[15]
            corru_x = inputs[16]
            corru_y = inputs[17]
            
            seq1 = inputs[18]
            x_seq1 = inputs[19]
            y_seq1 = inputs[20]
            position1 = inputs[21]
            x_position1 = inputs[22]
            y_position1 = inputs[23]
            ground1 = inputs[24]
            share_x_ground1 = inputs[25]
            share_y_ground1 = inputs[26]
            x_ground1 = inputs[27]
            y_ground1 = inputs[28]
            ground_mask1 = inputs[29]
            share_x_ground_mask1 = inputs[30]
            share_y_ground_mask1 = inputs[31]
            x_ground_mask1 = inputs[32]
            y_ground_mask1 = inputs[33]
            corru_x1 = inputs[34]
            corru_y1 = inputs[35]
            
        return seq, x_seq, y_seq, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, seq1, x_seq1, y_seq1, position1, x_position1, y_position1, ground1, share_x_ground1, share_y_ground1, x_ground1, y_ground1, ground_mask1, share_x_ground_mask1, share_y_ground_mask1, x_ground_mask1, y_ground_mask1, corru_x1, corru_y1

    def HingeLoss(self, pos, neg):
        gamma = torch.tensor(self.opt["margin"])
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()


    def train_batch(self, batch, args, x_feas, y_feas, epoch, i):
        self.model.train()
        self.optimizer.zero_grad()
        self.model.graph_convolution()

        seq, x_seq, y_seq, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, seq1, x_seq1, y_seq1, position1, x_position1, y_position1, ground1, share_x_ground1, share_y_ground1, x_ground1, y_ground1, ground_mask1, share_x_ground_mask1, share_y_ground_mask1, x_ground_mask1, y_ground_mask1, corru_x1, corru_y1 = self.unpack_batch(batch)

        # seq： 256 * 15， x_seq： 256 * 15， y_seq: 256 * 15， position: 256 * 15, x_position: 256 * 15, y_position: 256 * 15
        # ground:  256 * 10, share_x_ground: 256 * 10, share_y_ground: 256 * 10, x_ground: 256 * 10, y_ground: 256 * 10
       
        seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position)
        seqs_fea = self.model.projector(seqs_fea.reshape(seqs_fea.shape[0]*seqs_fea.shape[1],seqs_fea.shape[2])).reshape(seqs_fea.shape[0],seqs_fea.shape[1],seqs_fea.shape[2])
        x_seqs_fea = self.model.projector(x_seqs_fea.reshape(x_seqs_fea.shape[0]*x_seqs_fea.shape[1],x_seqs_fea.shape[2])).reshape(x_seqs_fea.shape[0],x_seqs_fea.shape[1],x_seqs_fea.shape[2])
        y_seqs_fea = self.model.projector(y_seqs_fea.reshape(y_seqs_fea.shape[0]*y_seqs_fea.shape[1],y_seqs_fea.shape[2])).reshape(y_seqs_fea.shape[0],y_seqs_fea.shape[1],y_seqs_fea.shape[2])
        
        seqs_fea1, x_seqs_fea1, y_seqs_fea1 = self.model(seq1, x_seq1, y_seq1, position1, x_position1, y_position1)
        seqs_fea1 = self.model.projector(seqs_fea1.reshape(seqs_fea1.shape[0]*seqs_fea1.shape[1],seqs_fea1.shape[2])).reshape(seqs_fea1.shape[0],seqs_fea1.shape[1],seqs_fea1.shape[2])
        x_seqs_fea1 = self.model.projector(x_seqs_fea1.reshape(x_seqs_fea1.shape[0]*x_seqs_fea1.shape[1],x_seqs_fea1.shape[2])).reshape(x_seqs_fea1.shape[0],x_seqs_fea1.shape[1],x_seqs_fea1.shape[2])
        y_seqs_fea1 = self.model.projector(y_seqs_fea1.reshape(y_seqs_fea1.shape[0]*y_seqs_fea1.shape[1],y_seqs_fea1.shape[2])).reshape(y_seqs_fea1.shape[0],y_seqs_fea1.shape[1],y_seqs_fea1.shape[2])
        
#         corru_x_fea = self.model.false_forward(corru_x, position)
#         corru_y_fea = self.model.false_forward(corru_y, position)

        x_mask = x_ground_mask.float().sum(-1).unsqueeze(-1).repeat(1,x_ground_mask.size(-1))
        x_mask = 1 / x_mask
        x_mask = x_ground_mask * x_mask # for mean
        x_mask = x_mask.unsqueeze(-1).repeat(1,1,seqs_fea.size(-1))
        s_x_fea = (x_seqs_fea * x_mask).sum(1)

        y_mask = y_ground_mask.float().sum(-1).unsqueeze(-1).repeat(1, y_ground_mask.size(-1))
        y_mask = 1 / y_mask
        y_mask = y_ground_mask * y_mask # for mean
        y_mask = y_mask.unsqueeze(-1).repeat(1,1,seqs_fea.size(-1))
        s_y_fea = (y_seqs_fea * y_mask).sum(1)
        
        c_x_fea = (seqs_fea * x_mask).sum(1)
        c_y_fea = (seqs_fea * y_mask).sum(1)

        x1 = (s_x_fea + c_x_fea)
        y1 = (s_y_fea + c_y_fea)
        
        x_mask1 = x_ground_mask1.float().sum(-1).unsqueeze(-1).repeat(1,x_ground_mask1.size(-1))
        x_mask1 = 1 / x_mask1
        x_mask1 = x_ground_mask1 * x_mask1 # for mean
        x_mask1 = x_mask1.unsqueeze(-1).repeat(1,1,seqs_fea1.size(-1))
        s_x_fea1 = (x_seqs_fea1 * x_mask1).sum(1)

        y_mask1 = y_ground_mask1.float().sum(-1).unsqueeze(-1).repeat(1, y_ground_mask1.size(-1))
        y_mask1 = 1 / y_mask1
        y_mask1 = y_ground_mask1 * y_mask1 # for mean
        y_mask1 = y_mask1.unsqueeze(-1).repeat(1,1,seqs_fea1.size(-1))
        s_y_fea1 = (y_seqs_fea1 * y_mask1).sum(1)
        
        c_x_fea1 = (seqs_fea1 * x_mask1).sum(1)
        c_y_fea1 = (seqs_fea1 * y_mask1).sum(1)

        x2 = (s_x_fea1 + c_x_fea1)
        y2 = (s_y_fea1 + c_y_fea1)
        
        p1 = self.model.predictor_x(x1)
        p2 = self.model.predictor_x(x2)
        q1 = self.model.predictor_y(y1)
        q2 = self.model.predictor_y(y2)

        loss_cl = -(self.criterion(p1, x2.detach()).mean() + self.criterion(p2, x1.detach()).mean()) * 0.5
        
        loss_cl += -(self.criterion(q1, y2.detach()).mean() + self.criterion(q2, y1.detach()).mean()) * 0.5
             
        if epoch > 0:
            server_x = x_feas[i]
            server_y = y_feas[i]
            loss_reg = -(self.criterion(x1, server_x).mean() + self.criterion(y1, server_y).mean())

        used = 10
        ground = ground[:,-used:]
        ground_mask = ground_mask[:, -used:]
        share_x_ground = share_x_ground[:, -used:]  # 256 * 10 item_id
        share_x_ground_mask = share_x_ground_mask[:, -used:]  # 256 * 10
        share_y_ground = share_y_ground[:, -used:] # 256 * 10
        share_y_ground_mask = share_y_ground_mask[:, -used:] # 256 * 10
        x_ground = x_ground[:, -used:]
        x_ground_mask = x_ground_mask[:, -used:]
        y_ground = y_ground[:, -used:]
        y_ground_mask = y_ground_mask[:, -used:]

        single_x_result =  self.model.lin_X(x_seqs_fea[:,-used:]) # b * seq * X_num(29208) seq = 10
        single_y_result = self.model.lin_Y(y_seqs_fea[:, -used:])  # b * seq * Y_num(34887)
        single_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
        single_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])  # b * seq * 1
        single_trans_x_result = torch.cat((single_x_result, single_x_pad_result), dim=-1) # b * seq * (X_num + 1)
        single_trans_y_result = torch.cat((single_y_result, single_y_pad_result), dim=-1) # b * seq * (Y_num + 1)
        
        x_single_loss = self.CS_criterion(
            single_trans_x_result.reshape(-1, self.opt["source_item_num"] + 1),  # (b * seq) * (X_num + 1)
            x_ground.reshape(-1))  # （b*seq， ）
        
        y_single_loss = self.CS_criterion(
            single_trans_y_result.reshape(-1, self.opt["target_item_num"] + 1),
            y_ground.reshape(-1))  # （b * seq， ）
        
        
        # seqs_fea[:, -used: ]从 256 * 15 * 256中提取出256 * 10 * 256（后10个，也就是预测出来的下一个tokens）
        share_x_result =  self.model.lin_X(seqs_fea[:,-used:]) # b * seq * X_num(29208) seq = 10
        share_y_result = self.model.lin_Y(seqs_fea[:, -used:])  # b * seq * Y_num(34887)
        share_pad_result = self.model.lin_PAD(seqs_fea[:, -used:])  # b * seq * 1
        share_trans_x_result = torch.cat((share_x_result, share_pad_result), dim=-1) # b * seq * (X_num + 1)
        share_trans_y_result = torch.cat((share_y_result, share_pad_result), dim=-1) # b * seq * (Y_num + 1)

        specific_x_result = self.model.lin_X(seqs_fea[:,-used:] + x_seqs_fea[:, -used:])  # b * seq * X_num
        specific_x_pad_result = self.model.lin_PAD(x_seqs_fea[:, -used:])  # b * seq * 1
        specific_x_result = torch.cat((specific_x_result, specific_x_pad_result), dim=-1) # b * seq * (X_num + 1)

        specific_y_result = self.model.lin_Y(seqs_fea[:,-used:] + y_seqs_fea[:, -used:])  # b * seq * Y_num
        specific_y_pad_result = self.model.lin_PAD(y_seqs_fea[:, -used:])  # b * seq * 1
        specific_y_result = torch.cat((specific_y_result, specific_y_pad_result), dim=-1) # b * seq * (Y_num + 1)

        x_share_loss = self.CS_criterion(
            share_trans_x_result.reshape(-1, self.opt["source_item_num"] + 1),  # (b * seq) * (X_num + 1)
            share_x_ground.reshape(-1))  
        # （b*seq， ）
        
        y_share_loss = self.CS_criterion(
            share_trans_y_result.reshape(-1, self.opt["target_item_num"] + 1),
            share_y_ground.reshape(-1))  # （b * seq， ）

        x_loss = self.CS_criterion(
            specific_x_result.reshape(-1, self.opt["source_item_num"] + 1),
            x_ground.reshape(-1))  # （b * seq， ）
        y_loss = self.CS_criterion(
            specific_y_result.reshape(-1, self.opt["target_item_num"] + 1),
            y_ground.reshape(-1))  # （b * seq， ）

        x_share_loss = (x_share_loss * (share_x_ground_mask.reshape(-1))).mean()
        y_share_loss = (y_share_loss * (share_y_ground_mask.reshape(-1))).mean()
        x_loss = (x_loss * (x_ground_mask.reshape(-1))).mean()
        y_loss = (y_loss * (y_ground_mask.reshape(-1))).mean()
        
        x_single_loss = (x_single_loss * (x_ground_mask.reshape(-1))).mean()
        y_single_loss = (y_single_loss * (y_ground_mask.reshape(-1))).mean()
        
#        loss = self.opt["lambda"] * (x_share_loss + y_share_loss + x_loss + y_loss + x_single_loss + y_single_loss) + (1 - self.opt["lambda"]) * loss_cl
        
        loss = x_share_loss + y_share_loss + x_loss + y_loss + x_single_loss + y_single_loss

        
#         if args.fed_method == "FedProx":
# #             loss += self.prox_reg(dict(self.model.named_parameters()), global_vars, args.mu)
#               loss += loss_reg * 0.01

#         if epoch > 0:
#             loss += loss_reg * 1.5
            
        loss.backward()
        self.optimizer.step()

        return loss.item(), x1.detach(), y1.detach()
    
    @ staticmethod
    def flatten(source):
        return torch.cat([value.flatten() for value in source])
    
    def prox_reg(self, params1, params2, mu):
        params2 = [params2[key] for key in params1.keys()]
        
        s1 = self.flatten(params1.values()) # 多维参数要用flatten压成一维的
        s2 = self.flatten(params2)
        return mu/2 * torch.norm(s1 - s2)

    def test_batch(self, batch):
        # x_seq: tensor([[64093,   113,   114,  ...,   119, 64093,   120],
        # [64093, 64093,   188,  ..., 64093, 64093, 64093],
        # [64093, 64093, 64093,  ..., 64093, 64093,   205],
        # ...,
        # [64093,  4080, 64093,  ..., 64093, 64093,  3064],
        # [64093, 64093, 64093,  ..., 64093, 64093, 64093],
        # [64093, 64093, 64093,  ..., 17827, 13222, 64093]], device='cuda:0')

        # y_seq: tensor([[64093, 64093, 64093,  ..., 64093, 29315, 64093],
        # [64093, 64093, 64093,  ..., 29392, 29393, 29394],
        # [64093, 64093, 64093,  ..., 29408, 29409, 64093],
        # ...,
        # [64093, 64093, 47027,  ..., 35250, 53940, 64093],
        # [64093, 64093, 64093,  ..., 34732, 53954, 47415],
        # [64093, 37068, 37070,  ..., 64093, 64093, 41327]], device='cuda:0')
        # X_or_Y: (840, )tensor([1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1...,
        # ground_truth: tensor([  109,   191,   203,   243,   252,   324,   475,   504,    63,   424,
        #   790,   695,   761,   767,   841,  1030,  1071,  1084,  1137,  1215,..)]       
        # tensor([[18577, 13593, 18518,  ..., 24936, 10670,   600],
        # [25703, 12733, 21206,  ...,   674,  4658, 12063],
        # [14734, 26148,  3060,  ..., 17719, 31894,  6911],
        # ...,
        # [19501, 19073,   477,  ..., 16528,  7572, 15010],
        # [32565, 11142, 28325,  ..., 19464, 27477, 29907],
        # [ 2891, 11400, 24322,  ...,  8345, 26233,  5031]], device='cuda:0')
        
        seq, x_seq, y_seq, position, x_position, y_position, X_last, Y_last, XorY, ground_truth, neg_list = self.unpack_batch_predict(batch)
        # seq: (840, 15), x_seq: (840, 15), y_seq: (840, 15)
        # position: (840, 15), x_position: (840, 15), y_position: (840, 15), 
        # X_last: (840, ) Y_last: (840, )(最后一个元素索引集合) X_or_Y: (840, )
        # ground_truth: (840,) neg_list: (840, 999)
        seqs_fea, x_seqs_fea, y_seqs_fea = self.model(seq, x_seq, y_seq, position, x_position, y_position)
        
        seqs_fea = self.model.projector(seqs_fea.reshape(seqs_fea.shape[0]*seqs_fea.shape[1],seqs_fea.shape[2])).reshape(seqs_fea.shape[0],seqs_fea.shape[1],seqs_fea.shape[2])
        
        x_seqs_fea = self.model.projector(x_seqs_fea.reshape(x_seqs_fea.shape[0]*x_seqs_fea.shape[1],x_seqs_fea.shape[2])).reshape(x_seqs_fea.shape[0],x_seqs_fea.shape[1],x_seqs_fea.shape[2])
        
        y_seqs_fea = self.model.projector(y_seqs_fea.reshape(y_seqs_fea.shape[0]*y_seqs_fea.shape[1],y_seqs_fea.shape[2])).reshape(y_seqs_fea.shape[0],y_seqs_fea.shape[1],y_seqs_fea.shape[2])
        
#         x_fea = torch.matmul(x_seqs_fea, self.model.attention1) + torch.matmul(seqs_fea, 1 - self.model.attention1)
#         y_fea = torch.matmul(y_seqs_fea, self.model.attention2) + torch.matmul(seqs_fea, 1 - self.model.attention2)
        # (840, 15, 256)  (840, 15, 256), (840, 15, 256)
        # 训练的时候是提取出后10个：256 * 10 * 256
        
        X_pred = []
        Y_pred = []
       
        for id, fea in enumerate(seqs_fea): # b * s * f
            # fea: (15, 256)
            if XorY[id] == 0: # 如果ground truth是领域X
                share_fea = seqs_fea[id, -1]
                specific_fea = x_seqs_fea[id, X_last[id]]
                X_score = self.model.lin_X(share_fea + specific_fea).squeeze(0)
                cur = X_score[ground_truth[id]]
                score_larger = (X_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy()
                true_item_rank = np.sum(score_larger) + 1
                X_pred.append(true_item_rank)

            else : # 如果ground truth是领域Y
                share_fea = seqs_fea[id, -1] # (256,) 取最后一个预测的token
                specific_fea = y_seqs_fea[id, Y_last[id]] # (256, )
                Y_score = self.model.lin_Y(share_fea + specific_fea).squeeze(0) # (34886, )
                cur = Y_score[ground_truth[id]] # tensor(-0.2955) # ground truth 所对应物品的预测分
                score_larger = (Y_score[neg_list[id]] > (cur + 0.00001)).data.cpu().numpy() # (999, )
                # neg_list: (840, 999) tensor([[18577, 13593, 18518,  ..., 24936, 10670,   600],...)
                # score_larger: (999, ) [True, True, False, True, , ...]


                true_item_rank = np.sum(score_larger) + 1 # 638
                Y_pred.append(true_item_rank)

        return X_pred, Y_pred