import argparse
import os
import time
import numpy as np
import torch
from Dataset import TrainingDataset, VTDataset, data_load
from Model4 import Net
from torch.utils.data import DataLoader
from Train import train
from Full_vt import full_vt
from Val import validate
from Test import test
# from torch.utils.tensorboard import SummaryWriter
###############################248###########################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--data_path', default='movielens', help='Dataset path')
    
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')

    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=20, help='Workers number.')

    parser.add_argument('--dim_x', type=int, default=64, help='Dimension of embedding')
    parser.add_argument('--dim_latent', type=int, default=None, help='Dimension of embedding')
    parser.add_argument('--topK', type=int, default=10, help='Workers number.')
    parser.add_argument('--step', type=int, default=None, help='Step in full ranking.')

    parser.add_argument('--self_loop', default='False', help='Add self loop in model.')
    parser.add_argument('--has_act', default='True', help='Add activation function.')
    parser.add_argument('--has_trans', default='False', help='Add transform function.')
    parser.add_argument('--has_weight', default='True', help='Add weight in GCN.')
    parser.add_argument('--has_id', default='True', help='Add id_embedding in model.')
    parser.add_argument('--has_v', default='True', help='Have visual information.')
    parser.add_argument('--has_a', default='True', help='Have acoustic information.')
    parser.add_argument('--has_t', default='True', help='Have textual information.')
    
    parser.add_argument('--prefix', default='', help='Prefix of save_file')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--save_file', default='result.txt', help='File saving path.')
    args = parser.parse_args()
    
    seed = args.seed
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    ##########################################################################################################################################
    data_path = args.data_path
    learning_rate = args.l_r
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    topK = args.topK
    step = args.step
    prefix = args.prefix
    save_file = args.save_file

    self_loop = True if args.self_loop == 'True' else False
    has_act = True if args.has_act == 'True' else False
    has_trans = True if args.has_trans == 'True' else False
    has_weight = True if args.has_weight == 'True' else False
    has_id = True if args.has_id == 'True' else False
    has_v = True if args.has_v == 'True' else False
    has_a = True if args.has_a == 'True' else False
    has_t = True if args.has_t == 'True' else False

    pretrain = False
    dim_latent = args.dim_latent
    dim_x = args.dim_x
    writer = None#SummaryWriter()
    # with open(data_path+'/result/result{0}_{1}.txt'.format(l_r, weight_decay), 'w') as save_file:
    #     save_file.write('---------------------------------lr: {0} \t Weight_decay:{1} ---------------------------------\r\n'.format(l_r, weight_decay))
    ##########################################################################################################################################
    print('Data loading ...')
    num_user, num_item, train_edge, val_edge, test_edge, item_adj, user_item_dict, v_feat, a_feat, t_feat, pos_row, pos_col = data_load(data_path)
    
    train_dataset = TrainingDataset(num_user, num_item, user_item_dict, train_edge)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    val_data = np.load('./Data/'+data_path+'/val_full.npy', allow_pickle=True)
    test_data = np.load('./Data/'+data_path+'/test_full.npy', allow_pickle=True)
    print('Data has been loaded.')

    v_feat = torch.tensor(v_feat).cuda() if has_v else None
    a_feat = torch.tensor(a_feat).cuda() if has_a else None
    t_feat = torch.tensor(t_feat).cuda() if has_t else None
    ##########################################################################################################################################
    model = Net(num_user, num_item, train_edge, item_adj, user_item_dict, v_feat, a_feat, t_feat, pos_row, pos_col, weight_decay, [32, 8, 4], self_loop, has_act, has_trans, has_weight, has_id, dim_x).cuda()
    ##########################################################################################################################################
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': learning_rate}])#, weight_decay=weight_decay)
    ##########################################################################################################################################
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    num_decreases = 0 
    for epoch in range(num_epoch):
        loss = train(epoch, len(train_dataset), train_dataloader, model, optimizer, batch_size, writer)
        if torch.isnan(loss):
            with open('./Data/'+data_path+'/'+save_file.format(data_path), 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} is Nan'.format(learning_rate, weight_decay))
            break

        val_precision, val_recall, val_ndcg = full_vt(epoch, model, val_data, 'Val', writer)
        test_precision, test_recall, test_ndcg = full_vt(epoch, model, test_data, 'Test', writer)

        if test_recall > max_recall:
            max_precision = test_precision
            max_recall = test_recall
            max_NDCG = test_ndcg
            num_decreases = 0
        else:
            if num_decreases > 20:
                with open('./Data/'+data_path+'/'+save_file.format(data_path), 'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} =====> Precision:{2} \t Recall:{3} \t NDCG:{4}\r\n'.
                                    format(learning_rate, weight_decay, max_precision, max_recall, max_NDCG))
                print('*'*20)
                print(model.result[:, 32+8:32+8+4])
                print('-'*20)
                print(model.result[:, 32+8+32+8+4:32+8+32+8+4+4])
                print('-'*20)
                print(model.result[:, 32+8+32+8+4+4+32+8:])
                print('*'*20)


                break
            else:
                num_decreases += 1


    # writer.close()