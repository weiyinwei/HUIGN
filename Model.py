import math
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from SAGEConv import SAGEConv
from DenseSAGEConv import DenseSAGEConv
from torch_geometric.utils import add_self_loops
###############################248###########################################

class FeaturePooling(torch.nn.Module):
    def __init__(self, feat, item_adj, layers, self_loop, has_act, has_trans, has_weight):
        super(FeaturePooling, self).__init__()
        self.feat = torch.tensor(feat, dtype=torch.float).cuda()
        self.dim_x = self.feat.size(1)
        self.item_adj = torch.FloatTensor(item_adj).cuda()
        self.layers = layers
        self.has_act = has_act
        self.weight_list = nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.rand(self.dim_x, cluster))) for cluster in layers])
        self.conv_layer_list = nn.ModuleList([DenseSAGEConv(self.dim_x, self.dim_x, has_weight=has_weight, self_loop=self_loop) for cluster in layers])
        self.conv_layer = DenseSAGEConv(self.dim_x, self.dim_x, has_weight=has_weight, self_loop=self_loop)
    def forward(self):
        out = torch.tensor([]).cuda()
        item_adj = self.item_adj
        x = self.feat
        pre_out = None

        entropy_loss = 0.0
        indepence_loss = 0.0 

        for index in range(len(self.layers)):
            conv_layer = self.conv_layer_list[index]
            x = conv_layer(x, item_adj)

            if self.has_act:
                x = F.leaky_relu(conv_layer(x, item_adj))

            weight = self.weight_list[index]
            temp_out = torch.matmul(x, weight)

            temp_out = torch.softmax(temp_out, dim=1)
            entropy_loss += (-temp_out * torch.log(temp_out+1e-15)).sum(dim=-1).mean()

            item_adj = torch.chain_matmul(temp_out.t(), item_adj, temp_out)

            if pre_out is not None:
                temp_out = torch.matmul(pre_out, temp_out)
            out = torch.cat((out, temp_out), dim=1)
            pre_out = temp_out
            x = weight.t()

            indepence_loss += torch.norm(torch.matmul(weight.t(), weight)-torch.eye(self.layers[index]).cuda(), p=2)/(self.layers[index]*self.layers[index])

        return out, entropy_loss, indepence_loss


class Net(torch.nn.Module):
    def __init__(self, num_user, num_item, edge_index, item_adj, user_item_dict, v_feat, a_feat, t_feat, pos_row, pos_col, reg_weight, layers, self_loop, has_act, has_trans, has_weight, has_id, dim_x):
        super(Net, self).__init__()
        self.dim_x = dim_x
        self.num_user = num_user
        self.num_item = num_item
        self.has_id = has_id

        self.user_item_dict = user_item_dict
        
        self.reg_weight = reg_weight
        self.pos_row = torch.LongTensor(pos_row)
        self.pos_col = torch.LongTensor(pos_col)-num_user
        self.weight = torch.tensor([[1.0],[-1.0]]).cuda()
        self.v_pooling = self.a_pooling = self.t_pooling = None

        num_modal = 0
        if v_feat is not None:
            self.v_pooling = FeaturePooling(v_feat, item_adj, layers, self_loop, has_act, has_trans, has_weight)
            num_modal += 1
        if a_feat is not None:
            self.a_pooling = FeaturePooling(a_feat, item_adj, layers, self_loop, has_act, has_trans, has_weight)
            num_modal += 1
        if t_feat is not None:
            self.t_pooling = FeaturePooling(t_feat, item_adj, layers, self_loop, has_act, has_trans, has_weight)
            num_modal += 1
        
        self.feat_dim = num_modal*torch.tensor(layers).sum().item()

        self.id_embedding = Parameter(nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))))

        self.user_preferences = Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.feat_dim))))

        if has_id:
            self.result = torch.rand((num_user+num_item, self.feat_dim+dim_x)).cuda()
        else:
            self.result = torch.rand((num_user+num_item, self.feat_dim)).cuda()


    def forward(self):
        self.rep = torch.tensor([]).cuda()
        
        entropy_loss = indepence_loss = torch.tensor([0.0]).cuda()

        if self.v_pooling is not None:
            self.v_rep, v_entropy_loss, v_indepence_loss = self.v_pooling()
            self.rep = torch.cat([self.rep, self.v_rep], dim=1)
            entropy_loss += v_entropy_loss
            v_indepence_loss += v_indepence_loss
        
        if self.a_pooling is not None:
            self.a_rep, a_entropy_loss, a_indepence_loss = self.a_pooling()
            self.rep = torch.cat([self.rep, self.a_rep], dim=1)
            entropy_loss += a_entropy_loss
            indepence_loss += a_indepence_loss

        if self.t_pooling is not None:
            self.t_rep, t_entropy_loss, t_indepence_loss = self.t_pooling()
            self.rep = torch.cat([self.rep, self.t_rep], dim=1)
            entropy_loss += t_entropy_loss
            indepence_loss += t_indepence_loss

        self.u_i_rep = torch.cat((self.user_preferences, self.rep), dim=0)

        if self.has_id:
            x = torch.cat((self.id_embedding, self.u_i_rep), dim=1)
        else:
            x = self.u_i_rep

        self.result = x
        return x, entropy_loss, indepence_loss

    def loss(self, user_tensor, item_tensor):
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1)
        out, entropy_loss, indepence_loss = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score*item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        if self.has_id:
            reg_embedding_loss = (self.id_embedding[user_tensor]**2).mean()+(self.id_embedding[item_tensor]**2).mean() + (self.user_preferences[user_tensor]**2).mean()
        else:
            reg_embedding_loss = (self.user_preferences[user_tensor]**2).mean()
        reg_loss = self.reg_weight * (reg_embedding_loss + entropy_loss + indepence_loss)

        return loss+reg_loss, loss, reg_embedding_loss, entropy_loss, indepence_loss

    def full_accuracy(self, val_data, step=2000, topk=10):
        user_tensor = self.result[:self.num_user]
        item_tensor = self.result[self.num_user:]

        start_index = 0
        end_index = self.num_user if step==None else step

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col))-self.num_user
                    score_matrix[row][col] = 1e-5

            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+self.num_user), dim=0)
            start_index = end_index
            
            if end_index+step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        length = len(val_data)        
        precision = recall = ndcg = 0.0

        for data in val_data:
            user = data[0]
            pos_items = set(data[1:])
            num_pos = len(pos_items)
            items_list = all_index_of_rank_list[user].tolist()

            items = set(items_list)

            num_hit = len(pos_items.intersection(items))
            
            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_hit, topk)):
                max_ndcg_score += 1 / math.log2(i+2)
            if max_ndcg_score == 0:
                continue
                
            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i+2)

            ndcg += ndcg_score/max_ndcg_score

        return precision/length, recall/length, ndcg/length

    # def full_accuracy(self, val_data, topk=10):
    #     user_tensor = self.result[:self.num_user]
    #     item_tensor = self.result[self.num_user:]

    #     score_matrix = torch.matmul(user_tensor, item_tensor.t())
    #     score_matrix[self.pos_row, self.pos_col] = -1e8
    #     precision = recall = ndcg = 0.0

    #     _, index_of_rank_list = torch.topk(score_matrix, topk)
    #     index_of_rank_list = index_of_rank_list.cpu()+self.num_user

    #     length = len(val_data)

    #     val_pbar = tqdm(total=length)
    #     log = math.log

    #     for data in val_data:
    #         val_pbar.update(1)
    #         user = data[0]
    #         pos_items = set(data[1:])
    #         num_pos = len(pos_items)
    #         items_list = index_of_rank_list[user].tolist()
    #         items = set(items_list)

    #         num_hit = len(pos_items.intersection(items))

    #         precision += float(num_hit / topk)
    #         recall += float(num_hit / num_pos)
    #         ndcg_score = 0.0
    #         max_ndcg_score = 0.0

    #         for i in range(min(num_hit, topk)):
    #             max_ndcg_score += 1 / math.log2(i+2)
    #         if max_ndcg_score == 0:
    #             continue
                
    #         for i, temp_item in enumerate(items_list):
    #             if temp_item in pos_items:
    #                 ndcg_score += 1 / math.log2(i+2)

    #         ndcg += ndcg_score/max_ndcg_score

    #     val_pbar.close()
    #     return precision/length, recall/length, ndcg/length