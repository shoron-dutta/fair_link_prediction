# For different values of lambda (regularizer parameter for controlling mutual information),
# generate plot for how accuracy, DM_m, EO_m changes

import torch
import os
import os.path as osp
import argparse
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from entropy import get_kernel_mat, get_normalized_kernel_mat, get_mutual_info

from utils import (
    get_link_labels,
    prediction_fairness,
    plot
)

from torch_geometric.utils import train_test_split_edges

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

from itertools import combinations_with_replacement
def fair_metrics(gt, y, group):
    metrics_dict = {
        "DPd": demographic_parity_difference(gt, y, sensitive_features=group),
        "EOd": equalized_odds_difference(gt, y, sensitive_features=group),
    }
    return metrics_dict

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[: pos_edge_index.size(1)] = 1.0
    return link_labels

def prediction_fairness(test_edge_idx, test_edge_labels, te_y, group):
    te_dyadic_src = group[test_edge_idx[0]]
    te_dyadic_dst = group[test_edge_idx[1]]

    # SUBGROUP DYADIC
    u = list(combinations_with_replacement(np.unique(group), r=2))

    te_sub_diatic = []
    for i, j in zip(te_dyadic_src, te_dyadic_dst):
        for k, v in enumerate(u):
            if (i, j) == v or (j, i) == v:
                te_sub_diatic.append(k)
                break
    te_sub_diatic = np.asarray(te_sub_diatic)
    # MIXED DYADIC 
    
    te_mixed_dyadic = te_dyadic_src != te_dyadic_dst
    # GROUP DYADIC
    te_gd_dict = fair_metrics(
        np.concatenate([test_edge_labels, test_edge_labels], axis=0),
        np.concatenate([te_y, te_y], axis=0),
        np.concatenate([te_dyadic_src, te_dyadic_dst], axis=0),
    )

    te_md_dict = fair_metrics(test_edge_labels, te_y, te_mixed_dyadic)

    te_sd_dict = fair_metrics(test_edge_labels, te_y, te_sub_diatic)

    fair_list = [
        te_md_dict["DPd"],
        te_md_dict["EOd"],
        te_gd_dict["DPd"],
        te_gd_dict["EOd"],
        te_sd_dict["DPd"],
        te_sd_dict["EOd"],
    ]

    return fair_list

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def encode(self, x, pos_edge_index):
        x = F.relu(self.conv1(x, pos_edge_index))
        x = self.conv2(x, pos_edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits, edge_index

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)



from torch_geometric.datasets import Planetoid
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':

    device = 'cpu'
    # print(f'Using {device} device')

    parser = argparse.ArgumentParser(description='Fair link prediction using regularized mutual information')
    parser.add_argument('--dataset', type=str, default='cora', help='name of the dataset: citeseer, pubmed, cora')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha parameter for entropy')
    args = parser.parse_args()

    dataset = args.dataset #"citeseer" #"cora" "pubmed"
    path = osp.join(osp.dirname(osp.realpath('__file__')), "..", "data", dataset)
    dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

    random_seed = 0
    acc_per_lambda = []
    dp_mix_per_lambda = []
    eo_mix_per_lambda = []
    # test_seeds = [0]
    acc_auc = []
    fairness = []


    delta = 0.16 # when set to 0, it applies no fairness constraint
    alpha = args.alpha
    reg_lambda_set = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for reg_lambda in reg_lambda_set:
    #for random_seed in test_seeds:

        np.random.seed(random_seed)
        data = dataset[0]
        protected_attribute = data.y
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.2)
        data = data.to(device)

        num_classes = len(np.unique(protected_attribute))
        N = data.num_nodes
        
        
        epochs = args.num_epochs
        model = GCN(data.num_features, 128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        

        Y = torch.LongTensor(protected_attribute).to(device)
        Y_aux = (
            Y[data.train_pos_edge_index[0, :]] != Y[data.train_pos_edge_index[1, :]]
        ).to(device)
        randomization = (
            torch.FloatTensor(epochs, Y_aux.size(0)).uniform_() < 0.5 + delta
        ).to(device)
        
        
        best_val_perf = test_perf = 0
        for epoch in tqdm(range(1, epochs)):
            
            # TRAINING    
            neg_edges_tr = negative_sampling(
                edge_index=data.train_pos_edge_index,
                num_nodes=N,
                num_neg_samples=data.train_pos_edge_index.size(1) // 2,
            ).to(device)
            # print(f'type(data.train_pos_edge_index): {type(data.train_pos_edge_index)}')
            # print(f'(train_pos_edge_index): {(data.train_pos_edge_index.shape)}, {data.train_pos_edge_index[:15]}')
            # print(f'data.x: {data.x.shape}, {data.y}')
            
            model.train()
            optimizer.zero_grad()

            z = model.encode(data.x, data.train_pos_edge_index)
            link_logits, _ = model.decode(
                z, data.train_pos_edge_index, neg_edges_tr
            )
            tr_labels = get_link_labels(
                data.train_pos_edge_index, neg_edges_tr
            ).to(device)
            
            # protected attributes of node pairs for each positive edge
            pos_edge_protected_attr =torch.empty(data.train_pos_edge_index.shape[1], 2)
            for i in range(data.train_pos_edge_index.shape[1]):
                src, tgt = data.train_pos_edge_index[0][i], data.train_pos_edge_index[1][i]
                pos_edge_protected_attr[i][0], pos_edge_protected_attr[i][1] = protected_attribute[src], protected_attribute[tgt]
            
            # protected attributes of node pairs for each negative edge
            neg_edge_protected_attr =torch.empty(neg_edges_tr.shape[1], 2)
            for i in range(neg_edges_tr.shape[1]):
                src, tgt = neg_edges_tr[0][i], neg_edges_tr[1][i]
                neg_edge_protected_attr[i][0], neg_edge_protected_attr[i][1] = protected_attribute[src], protected_attribute[tgt]

            node_pair_protected_attr = torch.cat((pos_edge_protected_attr, neg_edge_protected_attr), dim=0) # src and tgt class label for all edges, pos followed by neg
                
            
            # mutual info
            logit_arr = link_logits
            labels_arr = tr_labels
        
            
            
            # print(f'normalized_kernel_logits: {type(normalized_kernel_logits)}, normalized_kernel_sensitive: {type(normalized_kernel_sensitive)}')
            #print("Computed normalized kernel mat")
            mutual_info = get_mutual_info(logit_arr, final_vals, alpha)
            #print("Computed mutual info")


            loss = F.binary_cross_entropy_with_logits(link_logits, tr_labels) + reg_lambda * mutual_info
            loss.backward()
            optimizer.step()

            # EVALUATION
            model.eval()
            perfs = []
            for prefix in ["val", "test"]:
                pos_edge_index = data[f"{prefix}_pos_edge_index"]
                neg_edge_index = data[f"{prefix}_neg_edge_index"]
                with torch.no_grad():
                    z = model.encode(data.x, data.train_pos_edge_index)
                    link_logits, edge_idx = model.decode(z, pos_edge_index, neg_edge_index)
                link_probs = link_logits.sigmoid()
                link_labels = get_link_labels(pos_edge_index, neg_edge_index)
                auc = roc_auc_score(link_labels.cpu(), link_probs.cpu())
                perfs.append(auc)

            val_perf, tmp_test_perf = perfs
            if val_perf > best_val_perf:
                best_val_perf = val_perf
                test_perf = tmp_test_perf
            if epoch%10==0:
                log = "Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}"
                print(log.format(epoch, loss, best_val_perf, test_perf))

        # FAIRNESS
        auc = test_perf
        cut = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        best_acc = 0
        best_cut = 0.5
        for i in cut:
            
            acc = accuracy_score(link_labels.cpu(), link_probs.cpu() >= i)
            if acc > best_acc:
                best_acc = acc
                best_cut = i
        f = prediction_fairness(
            edge_idx.cpu(), link_labels.cpu(), link_probs.cpu() >= best_cut, Y.cpu()
        )
        print(f'best cut = {best_cut}')
        acc_auc.append([best_acc * 100, auc * 100])
        fairness.append([x * 100 for x in f])


        # since, we're using only one seed, mean is unnecessary but harmless, change later
        ma = np.mean(np.asarray(acc_auc), axis=0)
        mf = np.mean(np.asarray(fairness), axis=0)

        acc_per_lambda.append(ma[0])
        dp_mix_per_lambda.append(mf[0])
        eo_mix_per_lambda.append(mf[1])

    print(f'Accuracy: {acc_per_lambda}')
    print(f'DP (mixed): {dp_mix_per_lambda}')
    print(f'EO (mixed): {eo_mix_per_lambda}')
    
    fig_name = args.dataset + '_ablation.jpg'
    plot(acc_per_lambda, 'Accuracy', dp_mix_per_lambda, 'DP (mixed)', eo_mix_per_lambda, 'EO (mixed)', 'Performance metrics', fig_name)
    