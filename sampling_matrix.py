import torch

class SamplingMatrix:
    def __init__(self, g0, gX, adj, sample_size=3):
        self.g0_ratio = torch.tensor(1)
        self.gX_ratio = torch.tensor(1)
        self.g0gX_ratio = torch.tensor(1)
        self.sample_size = sample_size

        self.g0 = g0
        self.g0_idx = (g0.nonzero()).t()
        self.gX = gX
        self.gX_idx = (gX.nonzero()).t()

        self.g0_sampling = torch.zeros_like(adj)
        self.g0_sampling.index_fill_(0, (g0.nonzero()).squeeze(), 1)
        self.g0_sampling.index_fill_(1, (gX.nonzero()).squeeze(), 0)
        self.g0_sampling.fill_diagonal_(0)

        self.gX_sampling = torch.zeros_like(adj)
        self.gX_sampling.index_fill_(0, (gX.nonzero()).squeeze(), 1)
        self.gX_sampling.index_fill_(1, (g0.nonzero()).squeeze(), 0)
        self.gX_sampling.fill_diagonal_(0)

        self.g0gX_sampling = torch.ones_like(adj)
        self.g0gX_sampling -= self.g0_sampling + self.gX_sampling
        self.g0gX_sampling.fill_diagonal_(0)

        self.updateSamplingMatrix()

    def getRatio(self, verbose=True):
        """
        g0, gX, g0gX
        """
        total = self.g0_ratio + self.gX_ratio + self.g0gX_ratio
        g0_r = self.g0_ratio / total
        gX_r = self.gX_ratio / total
        g0gX_r = self.g0gX_ratio / total
        if verbose:
            print(f"G0:  {g0_r:.2%}")
            print(f"GX:  {gX_r:.2%}")
            print(f"G0GX:{g0gX_r:.2%}")

        return g0_r, gX_r, g0gX_r
    
    def updateSamplingMatrix(self):
        total = self.g0_ratio + self.gX_ratio + self.g0gX_ratio
        constant_g0 =       ((self.g0_ratio * self.sample_size * 2) / (total * self.g0_sampling.sum()))
        constant_gX =       ((self.gX_ratio * self.sample_size * 2) / (total * self.gX_sampling.sum()))
        constant_g0gX =    ((self.g0gX_ratio * self.sample_size * 2) / (total * self.g0gX_sampling.sum()))

        self.sampling_matrix = \
            self.g0gX_sampling * constant_g0gX + \
            self.g0_sampling * constant_g0 + \
            self.gX_sampling * constant_gX

        self.sampling_matrix = torch.clamp(self.sampling_matrix, min=0, max=1)

        self.sampling_matrix.triu_(diagonal=1)
        # self.sampling_matrix.fill_diagonal_(0)

    def get_sample(self):
        idx = torch.bernoulli(self.sampling_matrix)
        sampled = torch.triu(idx).float().nonzero().permute(1, 0)
        return sampled

    def get_sample_pairs(self):
        """
        returns (sampled idx, reverse mapping)
        """
        idx = torch.bernoulli(self.sampling_matrix)
        sampled = idx.nonzero().permute(1, 0)
        nz = (idx.sum(dim=1) != 0) + (idx.sum(dim=0) != 0)
        reverse = idx[nz, :][:, nz].nonzero().permute(1, 0)

        return sampled, reverse
    
    def updateRatio(self, g0_ratio, gX_ratio, g0gX_ratio):
        self.g0_ratio = g0_ratio
        self.gX_ratio = gX_ratio
        self.g0gX_ratio = g0gX_ratio
        total = self.g0_ratio + self.gX_ratio + self.g0gX_ratio
        self.g0_ratio /= total
        self.gX_ratio /= total
        self.g0gX_ratio /= total

        self.updateSamplingMatrix()

    def updateByGrad(self, adj_grad, count):
        min_sample = self.sample_size / 10

        g0r_count = (count * self.g0_sampling).sum() + min_sample
        gXr_count = (count * self.gX_sampling).sum() + min_sample
        g0gXr_count = count.sum() - (g0r_count + gXr_count) + min_sample

        abs_grad = adj_grad.abs()
        g0r = (abs_grad * self.g0_sampling).sum() / g0r_count
        gXr = (abs_grad * self.gX_sampling).sum() / gXr_count
        g0gXr = (abs_grad.sum() - (g0r + gXr)) / g0gXr_count

        total = g0r + gXr + g0gXr
        g0r /= total
        gXr /= total
        g0gXr /= total

        self.updateRatio(
            g0_ratio=(self.g0_ratio + g0r) / 2, 
            gX_ratio=(self.gX_ratio + gXr) / 2, 
            g0gX_ratio=(self.g0gX_ratio + g0gXr) / 2
        )


class SamplingMatrix2:
    def __init__(self, g0, gX, adj, sample_size=3):
        self.g0_ratio = torch.tensor(1)
        self.gX_ratio = torch.tensor(1)
        self.g0gX_ratio = torch.tensor(1)
        self.sample_size = sample_size

        self.updateSamplingMatrix()

    def getRatio(self, verbose=True):
        """
        g0, gX, g0gX
        """
        total = self.g0_ratio + self.gX_ratio + self.g0gX_ratio
        g0_r = self.g0_ratio / total
        gX_r = self.gX_ratio / total
        g0gX_r = self.g0gX_ratio / total
        if verbose:
            print(f"G0:  {g0_r:.2%}")
            print(f"GX:  {gX_r:.2%}")
            print(f"G0GX:{g0gX_r:.2%}")

        return g0_r, gX_r, g0gX_r
    
    def get_sample(self):
        idx = torch.bernoulli(self.sampling_matrix)
        sampled = torch.triu(idx).float().nonzero().permute(1, 0)
        return sampled

    def get_sample_pairs(self):
        """
        returns (sampled idx, reverse mapping)
        """
        idx = torch.bernoulli(self.sampling_matrix)
        sampled = idx.nonzero().permute(1, 0)
        nz = (idx.sum(dim=1) != 0) + (idx.sum(dim=0) != 0)
        reverse = idx[nz, :][:, nz].nonzero().permute(1, 0)

        return sampled, reverse
    
    def updateRatio(self, g0_ratio, gX_ratio, g0gX_ratio):
        self.g0_ratio = g0_ratio
        self.gX_ratio = gX_ratio
        self.g0gX_ratio = g0gX_ratio
        total = self.g0_ratio + self.gX_ratio + self.g0gX_ratio
        self.g0_ratio /= total
        self.gX_ratio /= total
        self.g0gX_ratio /= total

        self.updateSamplingMatrix()

    def updateByGrad(self, adj_grad, count):
        min_sample = self.sample_size / 10

        g0r_count = (count * self.g0_sampling).sum() + min_sample
        gXr_count = (count * self.gX_sampling).sum() + min_sample
        g0gXr_count = count.sum() - (g0r_count + gXr_count) + min_sample

        abs_grad = adj_grad.abs()
        g0r = (abs_grad * self.g0_sampling).sum() / g0r_count
        gXr = (abs_grad * self.gX_sampling).sum() / gXr_count
        g0gXr = (abs_grad.sum() - (g0r + gXr)) / g0gXr_count

        total = g0r + gXr + g0gXr
        g0r /= total
        gXr /= total
        g0gXr /= total

        self.updateRatio(
            g0_ratio=(self.g0_ratio + g0r) / 2, 
            gX_ratio=(self.gX_ratio + gXr) / 2, 
            g0gX_ratio=(self.g0gX_ratio + g0gXr) / 2
        )