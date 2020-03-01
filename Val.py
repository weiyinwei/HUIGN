from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import no_grad


def validate(epoch, length, dataloader, model, writer=None, batch_size=1):   
    print('Validation start...')
    model.eval()

    sum_pre = 0.0
    sum_recall = 0.0
    sum_ndcg_score = 0.0
    num_val = 0
    
    with no_grad():
        val_pbar = tqdm(total=length)
        for user_tensor, item_tensor in dataloader:
            val_pbar.update(batch_size)
            precision, recall, ndcg_score = model.accuracy(user_tensor, item_tensor)
            if recall < 0:
                continue
            sum_pre += precision
            sum_recall += recall
            sum_ndcg_score += ndcg_score
            num_val += batch_size
        val_pbar.close()

        print('---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
            epoch, sum_pre/num_val, sum_recall/num_val, sum_ndcg_score/num_val))
        
    # writer.add_scalar('val_Precition', sum_pre/num_val, epoch)
    # writer.add_scalar('val_Recall', sum_recall/num_val, epoch)
    # writer.add_scalar('val_NDCG', sum_ndcg_score/num_val, epoch)