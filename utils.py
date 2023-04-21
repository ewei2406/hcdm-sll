import torch
import dgl
import numpy as np
from deeprobust.graph.defense import GCN
from deeprobust.graph.data import Dataset
from deeprobust.graph.utils import preprocess


def load_data(dataset, seed, root='./tmp/'):
    clean_dataset = Dataset(root=root, name=dataset, seed=seed)
    adj, feat, labels = clean_dataset.adj, clean_dataset.features, clean_dataset.labels
    adj, feat, labels = preprocess(adj, feat, labels, preprocess_adj=False, device='cpu') # conver to tensor
    idx_train, idx_val, idx_test = clean_dataset.idx_train, clean_dataset.idx_val, clean_dataset.idx_test
    # adj = torch.tensor(clean_dataset.adj.toarray(), dtype=torch.float).to(device)
    # feat = torch.tensor(clean_dataset.features.toarray(), dtype=torch.float).to(device)
    # label = torch.tensor(clean_dataset.labels, dtype=torch.long).to(device)

    train_mask = torch.zeros([adj.shape[0]], dtype=torch.bool)  
    train_mask[idx_train] = 1
    test_mask = torch.zeros([adj.shape[0]], dtype=torch.bool)  
    test_mask[idx_test] = 1
    val_mask = torch.zeros([adj.shape[0]], dtype=torch.bool)  
    val_mask[idx_val] = 1

    return adj, feat, labels, train_mask, val_mask, test_mask


def get_g0(g0_size: int, A: torch.Tensor, method='random'):
    print('G0 with method: ' + method)
    def get_clusters(num_roots: int, max_hops: int, target_size: int) -> torch.Tensor:
        root_nodes = torch.rand(A.shape[0]).topk(num_roots).indices

        for hop in range(max_hops):
            newNodes = A[root_nodes].nonzero().t()[1]
            root_nodes = torch.cat((root_nodes, newNodes))
            root_nodes = torch.unique(root_nodes)
            if root_nodes.shape[0] >= target_size:
                break

        g0 = torch.zeros(A.shape[0])
        g0[root_nodes[:target_size]] = 1
        g0 = g0.bool()
        return g0

    g0 = torch.tensor([0])
    if method == 'many_clusters': # 10 nodes and their neighbors
        g0 = get_clusters(10, 10, g0_size)
    elif method == 'large_cluster': # 1 node and its neighbors
        g0 = get_clusters(1, 10, g0_size)
    elif method == 'random': # g0 is random/bias
        g0_probs = torch.ones(A.shape[0])
        g0_probs = g0_probs * (g0_size / g0_probs.sum())
        g0_probs.clamp_(0, 1)
        g0 = torch.bernoulli(g0_probs).bool()
    # elif args.g0_method == 'bias': # g0 is skewed toward a class by factor of 3
    #     bias = torch.randint(0, int(labels.max()) + 1, [1]).item()
    #     print(f'G0 class bias: {bias}')
    #     g0_probs = torch.ones(A.shape[0])
    #     g0_probs[labels == bias] = 3
    #     g0_probs = g0_probs * (g0_size / g0_probs.sum())
    #     g0_probs.clamp_(0, 1)
    #     g0 = torch.bernoulli(g0_probs).bool()

    print(f'G0 size: {g0.sum().item()}')
    print(f'G0 pct: {g0.sum().item() / A.shape[0]:.2%}')
    return g0


def evaluate(X, A, T, g, train_mask, val_mask, device='cpu'):
    X = X.to(device)
    A = A.to(device)
    θ = GCN(nfeat=X.shape[1], nclass=T.max().item()+1, nhid=32, device=device).to(device)
    masked = train_mask.clone()
    # masked[g] = 0
    θ.fit(X, A, T, masked, val_mask, train_iters=100)
    pred = θ(X, A).cpu()
    result = pred.argmax(1) == T
    acc_g0 = result[g].sum().item() / g.sum().item()
    acc_gX = result[~g].sum().item() / (~g).sum().item()
    print(f"G0: {acc_g0:.2%}")
    print(f"GX: {acc_gX:.2%}")
    return (acc_g0, acc_gX)


def scale(M: torch.Tensor, epsilon: int, patience=5) -> torch.Tensor:
    if M.abs().sum() == 0: return M
    
    for i in range(patience): # Maximum attempts
        if abs(epsilon / M.abs().sum() - 1) < 0.1: return M.clamp(-1, 1) # Stop with early convergence
        M = (M * (epsilon / M.abs().sum())).clamp(-1, 1)

    return M.clamp(-1, 1)

def discretize(M: torch.Tensor) -> torch.Tensor:
    return torch.bernoulli(M.abs().clamp(0, 1))

def truncate(M: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Truncates values in M such that only:
    1. positive values exist in M corresponding to non-existing edges
    2. negative values exist in M corresponding to alread existing edges
    """
    assert M.shape == A.shape
    negative_vals = (M * A).clamp(max=0)
    positive_vals = (M * (1-A)).clamp(min=0)
    return positive_vals + negative_vals

def xor(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A + B) - torch.mul(A * B, 2)

def binary(X) -> torch.Tensor:
    med = X.median()
    return (X > med).long()

def get_modified_adj(adj, perturbations):
    tri = (adj + perturbations) - torch.mul(adj * perturbations, 2)
    return tri

def idx_to_bool(idx, max_len=None):
    """
    Converts an array of indices into a boolean array (where desired indices are True)
    """
    
    if not max_len:
        max_len = max(idx) + 1
    arr = torch.zeros(max_len)
    arr[idx] = 1
    return arr > 0

def eval_acc(model, graph_feat, graph_adj, graph_labels, mask) -> float:
    pred = model(graph_feat, graph_adj).cpu().argmax(dim=1)
    acc = pred[mask] == graph_labels[mask].cpu()
    return (acc.sum() / acc.shape[0]).item()

def bin_feat(tensor: torch.Tensor, n_bins=50) -> torch.Tensor:
    """
    Discretizes a tensor by the number of bins
    """
    bins = []
    tensor_np = tensor.numpy()
    for i in range(n_bins):
        bins.append(np.percentile(tensor_np, 100 * i / n_bins))

    binned = torch.zeros_like(tensor)
    for i in range(0, n_bins):
        if i == 0:
            lower = 0
        else:
            lower = bins[i - 1]
        upper = bins[i]
        binned[(tensor < upper) * (tensor > lower)] = i

    return binned.long().to(tensor.device)

def projection(perturbations, n_perturbations):
    """
    Get the projection of a perturbation matrix such that the sum over the distribution of perturbations is n_perturbations 
    """
    def bisection(perturbations, a, b, n_perturbations, epsilon):
        def func(perturbations, x, n_perturbations):
            return torch.clamp(perturbations-x, 0, 1).sum() - n_perturbations
        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(perturbations, miu, n_perturbations) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(perturbations, miu, n_perturbations)*func(perturbations, a, n_perturbations) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu
    
    # projected = torch.clamp(self.adj_changes, 0, 1)
    if torch.clamp(perturbations, 0, 1).sum() > n_perturbations:
        left = (perturbations - 1).min()
        right = perturbations.max()
        miu = bisection(perturbations, left, right, n_perturbations, epsilon=1e-5)
        perturbations.data.copy_(torch.clamp(
            perturbations.data - miu, min=0, max=1))
    else:
        perturbations.data.copy_(torch.clamp(
            perturbations.data, min=0, max=1))
    
    return perturbations


def make_symmetric(adj):
    """
    Makes adj. matrix symmetric about the diagonal and sets the diagonal to 0.
    Keeps the upper triangle.
    """
    upper = torch.triu(adj)

    lower = torch.rot90(torch.flip(
        torch.triu(adj, diagonal=1), [0]), 3, [0, 1])

    result = (upper + lower).fill_diagonal_(0)
    return result


def calc_homophily(adj: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor=None) -> float:
    """
    returns H (number of similar edge / number of edges)
    """
    if mask != None:
        adj = adj[mask, :][:, mask]
        labels = labels[mask]
        
    edges = adj.nonzero().t()
    match = labels[edges[0]] == labels[edges[1]]

    if match.shape[0] == 0: return float('NaN')
    return match.sum().item() / match.shape[0]

def inner_homophily(adj, labels, g0, gX) -> float:
    """
    returns H between regions (number of similar edge / number of edges)
    """
    masked = adj.detach().clone()
    masked[g0, :][:, g0] = 0
    masked[gX, :][:, gX] = 0

    edges = masked.nonzero().t()
    match = labels[edges[0]] == labels[edges[1]]

    return match.sum().item() / match.shape[0]


def save_as_dgl(graph, adj, g0, name, root='./locked/'):
  edges = adj.to_sparse().indices()
  d = dgl.graph((edges[0], edges[1]), num_nodes=graph.num_nodes())
  d.ndata['g0'] = g0
  dgl.data.utils.save_graphs(f'{root}{name}.bin', [d], {"glabel": torch.tensor([0])})

def load_dgl(name, root='./locked/') -> dgl.DGLGraph:
  d = dgl.load_graphs(f'{root}{name}.bin')[0][0]
  return d