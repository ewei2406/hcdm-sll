import torch
import argparse
import numpy as np
import utils
import export

# Setup

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--method', type=str, default='SLL', choices=[
    'CER', 'REF', 'SLL', 'SLL_G'
])
parser.add_argument('--budget_pct', type=float, default=0.25)
parser.add_argument('--g0_method', type=str, default='random', choices=[
  'random', # randomly distribution of g0
  'large_cluster', # a random node and [g0_size] of its neighbors are in g0
  'many_clusters', # 10 random nodes and [g0_size] of their neighbors are in g0
  ])
parser.add_argument('--g0_size', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=10)
parser.add_argument('--T_s', type=int, default=1263)
parser.add_argument('--T_u', type=int, default=-1)
parser.add_argument('--dataset', type=str, default='cora', choices=[
    'Cora', 'Cora-ML', 'Citeseer', 'Pubmed', 'Polblogs', 'ACM', 'BlogCatalog', 'Flickr', 'UAI'
])
parser.add_argument('--ptb_rate', type=float, default=0.25)
args = parser.parse_args()

device = "cuda:1" if torch.cuda.is_available() else "cpu"
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Data

adj, feat, labels, train_mask, val_mask, test_mask = utils.load_data(args.dataset, args.seed)

T_s = T_u = torch.tensor([0])
if args.T_s == -1: 
    print("Sensitive task: label")
    T_s = labels
else:
    print("Sensitive task: " + str(args.T_s))
    T_s = utils.binary(feat[:,args.T_s].clone())
    feat[:,args.T_s] = 0
if args.T_u == -1: 
    print("Utility task: label")
    T_u = labels
else:
    print("Utility task: " + str(args.T_u))
    T_u = utils.binary(feat[:,args.T_u].clone())
    feat[:,args.T_u] = 0

g0_size = int(args.g0_size * adj.shape[0])
g0 = utils.get_g0(g0_size, adj, method=args.g0_method)

# Method

import attack
print("Attacking with method: " + args.method)
locked_adj = torch.zeros_like(adj)
if args.method == 'CER':
    locked_adj = attack.CER(adj, g0)
if args.method == 'REF':
    locked_adj = attack.CER(adj, g0)
if args.method == 'SLL':
    budget = (adj.shape[0]) * args.budget_pct
    locked_adj = attack.SLL(adj, feat, g0, T_s, T_u, budget, 100, lr=args.lr, train_mask=train_mask, val_mask=val_mask, device=device)
if args.method == 'SLL_G':
    budget = (adj.shape[0]) * args.budget_pct
    sample_size = int(adj.shape[0] * adj.shape[1] * 0.001)
    locked_adj = attack.SLL_G(adj, feat, g0, T_s, T_u, budget, sample_ct=5, sample_size=sample_size, epochs=30, lr=args.lr, train_mask=train_mask, val_mask=val_mask, device=device)

# Eval

print("Sensitive task:")
sens_g0, sens_gX = utils.evaluate(feat, locked_adj, T_s, g0, train_mask, val_mask, device)
print("Utility task:")
util_g0, util_gX = utils.evaluate(feat, locked_adj, T_u, g0, train_mask, val_mask, device)
edges_modified = (locked_adj.cpu() - adj).sum().item()
results = {
    "sens_g0": sens_g0,
    "sens_gX": sens_gX,
    "util_g0": util_g0,
    "util_gX": util_gX,
    "edges_modified": edges_modified
}
results.update(vars(args))
export.saveData('./results.csv', results)
