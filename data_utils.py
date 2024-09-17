from utils.loader import DataLoader
import os
import numpy as np
import torch
import random
import os
import torch
import torch.nn as nn
import random
import copy
import os
import torch
import torch.nn as nn
import numpy as np


def split_origin(n_clients, train_data, valid_data, test_data):
    # train_data_per_client = len(train_data)//n_clients
    # valid_data_per_client = len(valid_data)//n_clients
    # test_data_per_client = len(test_data)//n_clients
    # client_train_data = [ i for i in range(n_clients)]
    # client_valid_data = [ i for i in range(n_clients)]
    # client_test_data = [ i for i in range(n_clients)]

    # for c_id in range(n_clients):
    #     if c_id < n_clients - 1:
    #         client_train_data[c_id] = \
    #             train_data[c_id * train_data_per_client: (c_id + 1) * train_data_per_client]
    #         client_valid_data[c_id] = \
    #             valid_data[c_id * valid_data_per_client: (c_id + 1) * valid_data_per_client]  
    #         client_test_data[c_id] = \
    #             test_data[c_id * test_data_per_client: (c_id + 1) * test_data_per_client]  
    #     else:
    #         client_train_data[c_id] = train_data[c_id * train_data_per_client: ]
    #         client_valid_data[c_id] = valid_data[c_id * valid_data_per_client: ]
    #         client_test_data[c_id] = test_data[c_id * test_data_per_client: ]

    client_train_data = np.split(train_data, (np.cumsum([1/n_clients for i in range(n_clients)])[:-1]*len(train_data)).astype(int))
    client_valid_data = np.split(valid_data, (np.cumsum([1/n_clients for i in range(n_clients)])[:-1]*len(valid_data)).astype(int))
    client_test_data = np.split(test_data, (np.cumsum([1/n_clients for i in range(n_clients)])[:-1]*len(test_data)).astype(int))
    return client_train_data, client_valid_data, client_test_data


def split_quantity(n_clients, train_data, valid_data, test_data):
    fracs = [0.04569443, 0.29310702, 0.12821656, 0.0888967,  0.01651702, 0.01651424, 0.00582673, 0.19584123, 0.08949454, 0.11989152]
    #fracs = np.random.dirichlet([1]*n_clients)
    client_train_data = np.split(train_data, (np.cumsum(fracs)[:-1]*len(train_data)).astype(int))
    client_valid_data = np.split(valid_data, (np.cumsum(fracs)[:-1]*len(valid_data)).astype(int))   
    client_test_data = np.split(test_data, (np.cumsum(fracs)[:-1]*len(test_data)).astype(int))   
    return client_train_data, client_valid_data, client_test_data

def split_quality(n_clients, train_data, valid_data, test_data, train_seq_lens, valid_seq_lens, test_seq_lens):
    train_indices = np.array(train_seq_lens).argsort()
    valid_indices = np.array(valid_seq_lens).argsort()
    test_indices = np.array(test_seq_lens).argsort()
    
    train_data = np.array(train_data)[train_indices]
    valid_data = np.array(valid_data)[valid_indices]   
    test_data = np.array(test_data)[test_indices]

    client_train_data = np.split(train_data, (np.cumsum([1/n_clients for i in range(n_clients)])[:-1]*len(train_data)).astype(int))
    client_valid_data = np.split(valid_data, (np.cumsum([1/n_clients for i in range(n_clients)])[:-1]*len(valid_data)).astype(int))
    client_test_data = np.split(test_data, (np.cumsum([1/n_clients for i in range(n_clients)])[:-1]*len(test_data)).astype(int))

    return client_train_data, client_valid_data, client_test_data


def split_quantity_and_quality(n_clients, train_data, valid_data, test_data, train_seq_lens, valid_seq_lens, test_seq_lens):
    train_indices = np.array(train_seq_lens).argsort()
    valid_indices = np.array(valid_seq_lens).argsort()
    test_indices = np.array(test_seq_lens).argsort()
    
    train_data = np.array(train_data)[train_indices]
    valid_data = np.array(valid_data)[valid_indices]   
    test_data = np.array(test_data)[test_indices]

    client_train_data, client_valid_data, client_test_data = split_quantity(n_clients, train_data, valid_data, test_data)  
    return client_train_data, client_valid_data, client_test_data

def preprocess_dataset(opt):
    n_clients = opt["n_clients"]
    split = opt["split"]
    
    train_dataloader = DataLoader(opt['data_dir'], opt['batch_size'], opt, evaluation = -1)
    train_data = train_dataloader.data
    train_seq_lens = train_dataloader.seq_lens

    valid_dataloader = DataLoader(opt['data_dir'], opt["batch_size"], opt, evaluation = 2)
    valid_data = valid_dataloader.data
    valid_seq_lens = valid_dataloader.seq_lens

    test_dataloader = DataLoader(opt['data_dir'], opt["batch_size"], opt, evaluation = 1)
    test_data = test_dataloader.data
    test_seq_lens = test_dataloader.seq_lens

    if opt["split"] == "origin":
        client_train_data, client_valid_data, client_test_data = \
            split_origin(n_clients, train_data, valid_data, test_data)

    elif opt["split"] == "quantity":
        client_train_data, client_valid_data, client_test_data = \
            split_quantity(n_clients, train_data, valid_data, test_data)

    elif opt["split"] == "quality":
        client_train_data, client_valid_data, client_test_data = \
            split_quality(n_clients, train_data, valid_data, test_data, train_seq_lens, valid_seq_lens, test_seq_lens)
    elif opt["split"] == "both":
        client_train_data, client_valid_data, client_test_data = \
        split_quantity_and_quality(n_clients, train_data, valid_data, test_data, train_seq_lens, valid_seq_lens, test_seq_lens) 

    # print(len(client_train_data))
    # print(len(client_train_data[0])) # 804
    # print(len(client_train_data[0][0].shape)) 
    # print(len(client_train_data[1])) # 804
    # print(len(client_train_data[2])) # 804

    import pickle
    with open(os.path.join("dataset", opt['data_dir'], "client_train_data.pkl"), "wb") as f:
        pickle.dump(client_train_data, f)
    with open(os.path.join("dataset", opt['data_dir'], "client_valid_data.pkl"), "wb") as f:
        pickle.dump(client_valid_data, f)
    with open(os.path.join("dataset", opt['data_dir'], "client_test_data.pkl"), "wb") as f:
        pickle.dump(client_test_data, f)

def load_dataset(opt):
    import pickle

    with open(os.path.join("dataset", opt['data_dir'], "client_train_data.pkl"), "rb") as f:
        client_train_data = pickle.load(f)
    with open(os.path.join("dataset", opt['data_dir'], "client_valid_data.pkl"), "rb") as f:
        client_valid_data = pickle.load(f)
    with open(os.path.join("dataset", opt['data_dir'], "client_test_data.pkl"), "rb") as f:
        client_test_data = pickle.load(f)


    train_dataloader_list, valid_dataloader_list, test_dataloader_list = [], [], []
    for train_data, valid_data, test_data in zip(client_train_data, client_valid_data, client_test_data):
        train_dataloader = DataLoader.loadfromdata(train_data, opt['data_dir'], opt['batch_size'], opt, evaluation = -1)
        valid_dataloader = DataLoader.loadfromdata(valid_data, opt['data_dir'], opt['batch_size'], opt, evaluation = 2)
        test_dataloader = DataLoader.loadfromdata(test_data, opt['data_dir'], opt['batch_size'], opt, evaluation = 1)

        train_dataloader_list.append(train_dataloader)
        valid_dataloader_list.append(valid_dataloader)
        test_dataloader_list.append(test_dataloader)
        
    return train_dataloader_list, valid_dataloader_list, test_dataloader_list

def load_train_batch(dataloader, batch_size, source_item_num, target_item_num):
    
    data = dataloader.data
    data = preprocess(data, source_item_num, target_item_num)
#     print(data[0][6])
#     print(data[0][24])
#     print(data[0][9])
#     print(data[0][27])
#     print(data[0][10])
#     print(data[0][28])

#     exit()
    evaluation = dataloader.eval
    # shuffle for training
    if evaluation == -1:
        if batch_size > len(data):
            batch_size = len(data)
        if len(data) % batch_size != 0:
            data = np.concatenate([data, data[:batch_size]], axis=0)
        data = data[: (len(data)//batch_size) * batch_size]
    else :
        batch_size = 2048
    
    # chunk into batches
    batchfied_data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    for i in range(len(batchfied_data)):
        batch = batchfied_data[i]
        batch_size = len(batch)
        # print(batch_size)
        # print(batch[0].shape)
        if evaluation!=-1:
            batch = list(zip(*batch))
            yield (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]),torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]))
        else :
            batch = list(zip(*batch))
            yield (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]),torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]), torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]), torch.LongTensor(batch[18]), torch.LongTensor(batch[19]), torch.LongTensor(batch[20]), torch.LongTensor(batch[21]), torch.LongTensor(batch[22]), torch.LongTensor(batch[23]), torch.LongTensor(batch[24]), torch.LongTensor(batch[25]), torch.LongTensor(batch[26]), torch.LongTensor(batch[27]), torch.LongTensor(batch[28]), torch.LongTensor(batch[29]), torch.LongTensor(batch[30]), torch.LongTensor(batch[31]), torch.LongTensor(batch[32]), torch.LongTensor(batch[33]), torch.LongTensor(batch[34]), torch.LongTensor(batch[35]))

def load_batch(dataloader, batch_size):
    
    data = dataloader.data
    
    evaluation = dataloader.eval
    # shuffle for training
    if evaluation == -1:
        if batch_size > len(data):
            batch_size = len(data)
        if len(data) % batch_size != 0:
            data = np.concatenate([data, data[:batch_size]], axis=0)
        data = data[: (len(data)//batch_size) * batch_size]
    else :
        batch_size = 2048
    
    # chunk into batches
    batchfied_data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    for i in range(len(batchfied_data)):
        batch = batchfied_data[i]
        batch_size = len(batch)
        # print(batch_size)
        # print(batch[0].shape)
        if evaluation!=-1:
            batch = list(zip(*batch))
            yield (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]),torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]))
        else :
            batch = list(zip(*batch))
            yield (torch.LongTensor(batch[0]), torch.LongTensor(batch[1]), torch.LongTensor(batch[2]), torch.LongTensor(batch[3]),torch.LongTensor(batch[4]), torch.LongTensor(batch[5]), torch.LongTensor(batch[6]), torch.LongTensor(batch[7]),torch.LongTensor(batch[8]), torch.LongTensor(batch[9]), torch.LongTensor(batch[10]), torch.LongTensor(batch[11]), torch.LongTensor(batch[12]), torch.LongTensor(batch[13]), torch.LongTensor(batch[14]), torch.LongTensor(batch[15]), torch.LongTensor(batch[16]), torch.LongTensor(batch[17]))
            
            
def preprocess(data, source_item_num, target_item_num):

    processed = []

    max_len = 15


    for d in data: # the pad is needed! but to be careful.
        d1 = copy.deepcopy(d)
        ground = copy.deepcopy(d)[1:]
        share_x_ground = []
        share_x_ground_mask = []
        share_y_ground = []
        share_y_ground_mask = []
        for w in ground:
            if w < source_item_num:
                share_x_ground.append(w)
                share_x_ground_mask.append(1)
                share_y_ground.append(target_item_num)
                share_y_ground_mask.append(0)
            else:
                share_x_ground.append(source_item_num)
                share_x_ground_mask.append(0)
                share_y_ground.append(w - source_item_num)
                share_y_ground_mask.append(1)

        d = d[:-1]  # delete the ground truth
        position = list(range(len(d)+1))[1:]
        ground_mask = [1] * len(d)

        xd = []
        xcnt = 1
        x_position = []

        yd = []
        ycnt = 1
        y_position = []

        corru_x = []
        corru_y = []

        for w in d:
            if w < source_item_num:
                corru_x.append(w)
                xd.append(w)
                x_position.append(xcnt)
                xcnt += 1
                corru_y.append(random.randint(0, source_item_num - 1))
                yd.append(source_item_num + target_item_num)
                y_position.append(0)

            else:
                corru_x.append(random.randint(source_item_num, source_item_num + target_item_num - 1))
                xd.append(source_item_num + target_item_num)
                x_position.append(0)
                corru_y.append(w)
                yd.append(w)
                y_position.append(ycnt)
                ycnt += 1

        now = -1
        x_ground = [source_item_num] * len(xd) # caution!
        x_ground_mask = [0] * len(xd)
        for id in range(len(xd)):
            id+=1
            if x_position[-id]:
                if now == -1:
                    now = xd[-id]
                    if ground[-1] < source_item_num:
                        x_ground[-id] = ground[-1]
                        x_ground_mask[-id] = 1
                    else:
                        xd[-id] = source_item_num + target_item_num
                        x_position[-id] = 0
                else:
                    x_ground[-id] = now
                    x_ground_mask[-id] = 1
                    now = xd[-id]
        if sum(x_ground_mask) == 0:
            print("pass sequence x")
            continue

        now = -1
        y_ground = [target_item_num] * len(yd) # caution!
        y_ground_mask = [0] * len(yd)
        for id in range(len(yd)):
            id+=1
            if y_position[-id]:
                if now == -1:
                    now = yd[-id] - source_item_num
                    if ground[-1] > source_item_num:
                        y_ground[-id] = ground[-1] - source_item_num
                        y_ground_mask[-id] = 1
                    else:
                        yd[-id] = source_item_num + target_item_num
                        y_position[-id] = 0
                else:
                    y_ground[-id] = now
                    y_ground_mask[-id] = 1
                    now = yd[-id] - source_item_num
        if sum(y_ground_mask) == 0:
            print("pass sequence y")
            continue

        if len(d) < max_len:
            position = [0] * (max_len - len(d)) + position
            x_position = [0] * (max_len - len(d)) + x_position
            y_position = [0] * (max_len - len(d)) + y_position

            ground = [source_item_num + target_item_num] * (max_len - len(d)) + ground
            share_x_ground = [source_item_num] * (max_len - len(d)) + share_x_ground
            share_y_ground = [target_item_num] * (max_len - len(d)) + share_y_ground
            x_ground = [source_item_num] * (max_len - len(d)) + x_ground
            y_ground = [target_item_num] * (max_len - len(d)) + y_ground

            ground_mask = [0] * (max_len - len(d)) + ground_mask
            share_x_ground_mask = [0] * (max_len - len(d)) + share_x_ground_mask
            share_y_ground_mask = [0] * (max_len - len(d)) + share_y_ground_mask
            x_ground_mask = [0] * (max_len - len(d)) + x_ground_mask
            y_ground_mask = [0] * (max_len - len(d)) + y_ground_mask

            corru_x = [source_item_num + target_item_num] * (max_len - len(d)) + corru_x
            corru_y = [source_item_num + target_item_num] * (max_len - len(d)) + corru_y
            xd = [source_item_num + target_item_num] * (max_len - len(d)) + xd
            yd = [source_item_num + target_item_num] * (max_len - len(d)) + yd
            d = [source_item_num + target_item_num] * (max_len - len(d)) + d
        else:
            print("pass")

#             processed.append([d, xd, yd, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y])

        # 生成增强数据

        pre = copy.deepcopy(d1)[10:len(d1)-1]
        random.shuffle(pre)
        last = copy.deepcopy(d1)[len(d1)-1:]
        d1 = d1[0:10] + pre + last

        ground1 = copy.deepcopy(d1)[1:]
        share_x_ground1 = []
        share_x_ground_mask1 = []
        share_y_ground1 = []
        share_y_ground_mask1 = []
        for w in ground1:
            if w < source_item_num:
                share_x_ground1.append(w)
                share_x_ground_mask1.append(1)
                share_y_ground1.append(target_item_num)
                share_y_ground_mask1.append(0)
            else:
                share_x_ground1.append(source_item_num)
                share_x_ground_mask1.append(0)
                share_y_ground1.append(w - source_item_num)
                share_y_ground_mask1.append(1)

        d1 = d1[:-1]  # delete the ground truth
        position1 = list(range(len(d1)+1))[1:]
        ground_mask1 = [1] * len(d1)

        xd1 = []
        xcnt = 1
        x_position1 = []

        yd1 = []
        ycnt = 1
        y_position1 = []

        corru_x1 = []
        corru_y1 = []

        for w in d1:
            if w < source_item_num:
                corru_x1.append(w)
                xd1.append(w)
                x_position1.append(xcnt)
                xcnt += 1
                corru_y1.append(random.randint(0, source_item_num - 1))
                yd1.append(source_item_num + target_item_num)
                y_position1.append(0)

            else:
                corru_x1.append(random.randint(source_item_num, source_item_num + target_item_num - 1))
                xd1.append(source_item_num + target_item_num)
                x_position1.append(0)
                corru_y1.append(w)
                yd1.append(w)
                y_position1.append(ycnt)
                ycnt += 1

        now = -1
        x_ground1 = [source_item_num] * len(xd1) # caution!
        x_ground_mask1 = [0] * len(xd1)
        for id in range(len(xd1)):
            id+=1
            if x_position1[-id]:
                if now == -1:
                    now = xd1[-id]
                    if ground1[-1] < source_item_num:
                        x_ground1[-id] = ground1[-1]
                        x_ground_mask1[-id] = 1
                    else:
                        xd1[-id] = source_item_num + target_item_num
                        x_position1[-id] = 0
                else:
                    x_ground1[-id] = now
                    x_ground_mask1[-id] = 1
                    now = xd1[-id]

        if sum(x_ground_mask1) == 0:
            print("pass sequence x")
            continue

        now = -1
        y_ground1 = [target_item_num] * len(yd1) # caution!
        y_ground_mask1 = [0] * len(yd1)
        for id in range(len(yd1)):
            id+=1
            if y_position1[-id]:
                if now == -1:
                    now = yd1[-id] - source_item_num
                    if ground1[-1] > source_item_num:
                        y_ground1[-id] = ground1[-1] - source_item_num
                        y_ground_mask1[-id] = 1
                    else:
                        yd1[-id] = source_item_num + target_item_num
                        y_position1[-id] = 0
                else:
                    y_ground1[-id] = now
                    y_ground_mask1[-id] = 1
                    now = yd1[-id] - source_item_num
        if sum(y_ground_mask1) == 0:
            print("pass sequence y")
            continue

        if len(d1) < max_len:
            position1 = [0] * (max_len - len(d1)) + position1
            x_position1 = [0] * (max_len - len(d1)) + x_position1
            y_position1 = [0] * (max_len - len(d1)) + y_position1

            ground1 = [source_item_num + target_item_num] * (max_len - len(d1)) + ground1
            share_x_ground1 = [source_item_num] * (max_len - len(d1)) + share_x_ground1
            share_y_ground1 = [target_item_num] * (max_len - len(d1)) + share_y_ground1
            x_ground1 = [source_item_num] * (max_len - len(d1)) + x_ground1
            y_ground1 = [target_item_num] * (max_len - len(d1)) + y_ground1

            ground_mask1 = [0] * (max_len - len(d1)) + ground_mask1
            share_x_ground_mask1 = [0] * (max_len - len(d1)) + share_x_ground_mask1
            share_y_ground_mask1 = [0] * (max_len - len(d1)) + share_y_ground_mask1
            x_ground_mask1 = [0] * (max_len - len(d1)) + x_ground_mask1
            y_ground_mask1 = [0] * (max_len - len(d1)) + y_ground_mask1

            corru_x1 = [source_item_num + target_item_num] * (max_len - len(d1)) + corru_x1
            corru_y1 = [source_item_num + target_item_num] * (max_len - len(d1)) + corru_y1
            xd1 = [source_item_num + target_item_num] * (max_len - len(d1)) + xd1
            yd1 = [source_item_num + target_item_num] * (max_len - len(d1)) + yd1
            d1 = [source_item_num + target_item_num] * (max_len - len(d1)) + d1
        else:
            print("pass")

        processed.append([d, xd, yd, position, x_position, y_position, ground, share_x_ground, share_y_ground, x_ground, y_ground, ground_mask, share_x_ground_mask, share_y_ground_mask, x_ground_mask, y_ground_mask, corru_x, corru_y, d1, xd1, yd1, position1, x_position1, y_position1, ground1, share_x_ground1, share_y_ground1, x_ground1, y_ground1, ground_mask1, share_x_ground_mask1, share_y_ground_mask1, x_ground_mask1, y_ground_mask1, corru_x1, corru_y1])


    return processed