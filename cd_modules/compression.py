import torch

def decompose(A, top_sum=0.5):
    '''Low rank approximation of a 2D tensor, keeping only
    largest singular values that sums to topsum * (original sum)'''
    U, S, Vh = torch.linalg.svd(A.float(), full_matrices=False)
    r = max(1, sum(torch.cumsum(S, 0) < sum(S) * top_sum * 1.0001)) # 1.0001 because floats
    return U[:, :r] @ torch.diag(S)[:r, :r], Vh[:r]
