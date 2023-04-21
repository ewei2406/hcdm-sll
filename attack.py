import torch
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
from tqdm import tqdm
import utils

class SamplingMatrix:
    def __init__(self, num_nodes: int, g0: torch.Tensor) -> None:
        self.g0_idx = g0.nonzero()
        self.gX_idx = (~g0).nonzero()
        self.n = num_nodes
        self.r00 = 1/3
        self.r0X = 1/3
        self.rXX = 1/3
        pass

    def update_ratio(self, A_grad, ema_k=3):
        sum_00 = A_grad[self.g0_idx, self.g0_idx].sum()
        sum_XX = A_grad[self.gX_idx, self.gX_idx].sum()
        sum_0X = A_grad.sum() - sum_00 - sum_XX
        total = sum_00 + sum_0X + sum_XX
        self.r00 = ((ema_k - 1) * self.r00 +  (sum_00 / total)) / ema_k
        self.r0X = ((ema_k - 1) * self.r0X +  (sum_0X / total)) / ema_k
        self.rXX = ((ema_k - 1) * self.rXX +  (sum_XX / total)) / ema_k
        total = self.r00 + self.r0X + self.rXX
        self.r00 = self.r00 / total
        self.r0X = self.r0X / total
        self.rXX = self.rXX / total

    def get_sample(self, sample_size: int) -> torch.Tensor:
        num_00 = int(self.r00 * sample_size)
        g00 = torch.cat(
            [self.g0_idx[torch.randint(0, self.g0_idx.shape[0], [num_00])], 
             self.g0_idx[torch.randint(0, self.g0_idx.shape[0], [num_00])]]
             , 1)
        num_0X = int(self.r0X * sample_size)
        g0X = torch.cat(
            [self.g0_idx[torch.randint(0, self.g0_idx.shape[0], [num_0X])], 
             self.gX_idx[torch.randint(0, self.gX_idx.shape[0], [num_0X])]]
             , 1)
        num_XX = int(self.rXX * sample_size)
        gXX = torch.cat(
            [self.gX_idx[torch.randint(0, self.gX_idx.shape[0], [num_XX])], 
             self.gX_idx[torch.randint(0, self.gX_idx.shape[0], [num_XX])]]
             , 1)
        
        return torch.cat([g00, g0X, gXX], 0).t()


def SLL_G(
        A: torch.Tensor, 
        X:torch.Tensor, 
        g0: torch.Tensor, 
        T_s: torch.Tensor, 
        T_u: torch.Tensor, 
        ε: int,
        epochs: int,
        sample_ct: int,
        sample_size: int,
        lr: float,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        device='cpu') -> torch.Tensor:
    """
    T_s is sensitive classification task
    T_u is utility classification task
    g0 is array of boolean representing protected nodes.
    """
    gX = ~g0
    X = X.to(device)
    T_s = T_s.to(device)
    T_u = T_u.to(device)
    A = A.to(device)

    M = torch.zeros_like(A).float()
    θ_s = GCN(nfeat=X.shape[1], nclass=T_s.max().item()+1, nhid=32, device=device).to(device)
    θ_u = GCN(nfeat=X.shape[1], nclass=T_u.max().item()+1, nhid=32, device=device).to(device)

    t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    A_p = torch.zeros_like(A) # Initialize modified adj
    sampling_matrix = SamplingMatrix(A.shape[0], g0)

    for epoch in t:
        # A_p = xor(A, discretize(M, ε)).to(device).requires_grad_(True) # To clear graident of modified adj
        cts = torch.zeros_like(A, dtype=torch.int)
        A_grad = torch.zeros_like(A, dtype=torch.float)
        c_L = 0

        for _ in range(sample_ct):
            idx = sampling_matrix.get_sample(sample_size)

            sample = A_p[idx[0], idx[1]].clone().detach().requires_grad_(True).to(device)
            A_p[idx[0], idx[1]] = sample

            L = 0
            sens_pred = θ_s(X, A_p)
            utility_pred = θ_u(X, A_p)
            L += F.cross_entropy(sens_pred[g0], T_s[g0]) \
                - F.cross_entropy(sens_pred[gX], T_s[gX])
            L -= F.cross_entropy(utility_pred, T_u)

            grad = torch.autograd.grad(L, sample)[0]
            cts[idx[0], idx[1]] += 1
            A_grad[idx[0], idx[1]] += grad
            c_L += L.item()

        A_grad = torch.div(A_grad, cts)
        A_grad[A_grad != A_grad] = 0

        sampling_matrix.update_ratio(A_grad)
        M = utils.truncate(M + ((lr * A_grad) / (epoch + 1)), A)
        # M = scale(M, ε)
        M = utils.projection(M, ε)

        A_p = utils.xor(A, utils.discretize(M)).to(device)
        θ_s.fit(X, A_p, T_s, train_mask, val_mask, train_iters=1)
        θ_u.fit(X, A_p, T_u, train_mask, val_mask, train_iters=1)

        t.set_postfix({
            "loss": c_L,
            # "edges modified": (A_p.cpu() - A).abs().sum().item()
        })
    
    return A_p.requires_grad_(False)

def CER(A: torch.Tensor, g0: torch.Tensor) -> torch.Tensor:
    A = A.cpu()
    g0 = g0.cpu()
    A[:, g0] = 0
    A[g0, :] = 0
    return A

def REF(A: torch.Tensor, g0: torch.Tensor, ε: int) -> torch.Tensor:
    noise = torch.zeros_like(A)
    noise[g0, :] = 1
    noise[:, ~g0] = 0
    noise *= 2 * ε / noise.sum()
    noise = torch.bernoulli(noise.clamp(0, 1))
    return utils.xor(A, noise)

def SLL(
        A: torch.Tensor, 
        X:torch.Tensor, 
        g0: torch.Tensor, 
        T_s: torch.Tensor, 
        T_u: torch.Tensor, 
        ε: int,
        epochs: int,
        lr: float,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        surrogate_lr=1e-2,
        device='cpu') -> torch.Tensor:
    """
    T_s is sensitive classification task
    T_u is utility classification task
    g0 is array of boolean representing protected nodes.
    """
    gX = ~g0
    X = X.to(device)
    T_s = T_s.to(device)
    T_u = T_u.to(device)
    A = A.to(device)

    M = torch.zeros_like(A).float().to(device)
    θ_s = GCN(nfeat=X.shape[1], nclass=T_s.max().item()+1, nhid=32, lr=surrogate_lr, device=device).to(device)
    θ_u = GCN(nfeat=X.shape[1], nclass=T_u.max().item()+1, nhid=32, lr=surrogate_lr, device=device).to(device)

    t = tqdm(range(epochs), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    A_p = torch.zeros_like(A).to(device).requires_grad_(True) # Initialize modified adj

    for epoch in t:
        # A_p = xor(A, discretize(M, ε)).to(device).requires_grad_(True) # To clear graident of modified adj
        L = 0
        sens_pred = θ_s(X, A_p)
        L += F.cross_entropy(sens_pred[g0], T_s[g0]) \
            - F.cross_entropy(sens_pred[gX], T_s[gX])
        
        utility_pred = θ_u(X, A_p)
        L -= F.cross_entropy(utility_pred, T_u)

        A_grad = torch.autograd.grad(L, A_p)[0]
        M = utils.truncate(M + ((lr * A_grad) / (epoch + 1)), A)
        # M = scale(M, ε)
        M = utils.projection(M, ε)

        A_p = utils.xor(A, utils.discretize(M)).to(device).requires_grad_(True)
        θ_s.fit(X, A_p, T_s, train_mask, val_mask, train_iters=1)
        θ_u.fit(X, A_p, T_u, train_mask, val_mask, train_iters=1)

        t.set_postfix({
            "loss": L.item(),
            # "edges modified": (A_p.cpu() - A).abs().sum().item()
        })
    
    return A_p.requires_grad_(False)