#!/usr/bin/env python3
from turtle import forward
import torch
import torch.nn 

# Source (with alterations): https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51

def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Q, K, and V have same shape (batch_size, sequence_length, num_features)
        """
        QKt = Q.bmm(K.transpose(1, 2)) # batch matrix multiplication (bmm), only multiply "batch-by-batch"
        dk = Q.size(-1) ** 0.5
        attention_weights = torch.softmax(QKt / dk, dim=-1)
        return attention_weights.bmm(V)

class AttentionHead(torch.nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int) -> None:
        super().__init__()
        self._q = torch.nn.Linear(dim_in, dim_q)
        self._k = torch.nn.Linear(dim_in, dim_k)
        self._v = torch.nn.Linear(dim_in, dim_k)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return attention(self._q(Q), self._k(K), self._v(V))
        # return attention(Q, K, V)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int) -> None:
        super().__init__()
        self._heads = torch.nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self._linear = torch.nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self._linear(
            torch.cat([head(Q, K, V) for head in self._heads], dim=-1)
        )
        

class AudioSpectrogramTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.nn.Conv2d(1, )

    def forward(self, X):
        """X must have shape (batch_size, 1, frequency_bins, n_time_steps)"""
        batch_size = X.shape[0]
        return self.forward(X)


if __name__ == "__main__":
    t = torch.rand((12, 1024, 128))
    # print(t.shape)
    t = t.unsqueeze(1)
    # print(t.shape)
    t = t.transpose(2, 3)
    # print(t.shape)

    batch_size = 42
    num_features = 128
    sequence_length = 100 
    shape = (batch_size, sequence_length, num_features)
    Q, K, V = torch.rand(shape), torch.rand(shape), torch.rand(shape)

    m = AttentionHead(sequence_length, Q.size(-1), K.size(-1))
    A = m(Q.T, K.T, V)
    print(A.shape)