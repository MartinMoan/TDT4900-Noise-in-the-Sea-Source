#!/usr/bin/env python3
import torch
import torch.nn 

# Source (with alterations): https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51

def prepend_cls_token(input: torch.Tensor, cls_token: torch.Tensor) -> torch.Tensor:
    """Add cls_token to the beginning of every sequence in the input. 

    Args:
        input (torch.Tensor): input tensor with shape (batch_size, input_sequence_length, num_features)
        cls_token (torch.Tensor): classification token with shape (num_features,)

    Returns:
        torch.Tensor: tensor with cls_token prepended to every sequence in the input, output has shape (batch_size, input_sequence_length + 1, num_features)
    """
    cls = cls_token.repeat(batch_size, 1).unsqueeze(1)
    return torch.cat((cls, input), dim=1)

def attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Q, K, and V have same shape (batch_size, sequence_length, num_features)
    Returns:
        scaled dot product attention (torch.Tensor): a tensor with shape (batch_size, num_features, sequence_length)
    """
    QKt = Q.bmm(K.transpose(1, 2)) # batch matrix multiplication (bmm), only multiply "batch-by-batch"
    dk = Q.size(-1) ** 0.5
    attention_weights = torch.softmax(QKt / dk, dim=-1)
    return attention_weights.bmm(V)

def positional_encoding(sequence_length: int, dim_in: int, device: str = "cpu") -> torch.Tensor:
    pos = torch.arange(sequence_length, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_in, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** torch.div(dim, dim_in, rounding_mode="floor"))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_in: int = 512, dim_hidden: int = 2048) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(dim_in, dim_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_hidden, dim_in)
    )

class AddNormLayer(torch.nn.Module):
    def __init__(self, sublayer: torch.nn.Module, dim_out: int, dropout: float = 0.1) -> None:
        super().__init__()
        self._sublayer = sublayer
        self._norm = torch.nn.LayerNorm(dim_out)
        self._dropout = torch.nn.Dropout(dropout)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        return self._norm(tensors[0] + self._dropout(self._sublayer(*tensors)))

class AttentionHead(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self._in = dim_in
        self._out = dim_out
        self._q = torch.nn.Linear(dim_in, dim_out) # input (batch_size, seq_len, dim_in) output (batch_size, seq_len, dim_q)
        self._k = torch.nn.Linear(dim_in, dim_out) # input (batch_size, seq_len, dim_in) output (batch_size, seq_len, dim_k)
        self._v = torch.nn.Linear(dim_in, dim_out) # input (batch_size, seq_len, dim_in) output (batch_size, seq_len, dim_k)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Q, K and V with shape (batch_size, sequence_length, dim_in) returns a tensor with shape (batch_size, sequence_length, dim_out)
        """
        q = self._q(Q)
        k = self._k(K)
        v = self._v(V)
        return attention(q, k, v)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self._heads = torch.nn.ModuleList(
            [AttentionHead(dim_in, dim_out) for _ in range(num_heads)]
        )
        self._linear = torch.nn.Linear(num_heads * dim_out, dim_in)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self._linear(
            torch.cat([head(Q, K, V) for head in self._heads], dim=-1)
        )


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, dim_model: int = 512, num_heads: int = 6, dropout: float = 0.1, dim_feedforward: int = 2048) -> None:
        super().__init__()
        dim_in = max(dim_model // num_heads, 1)
        self._attention = AddNormLayer(
            sublayer=MultiHeadAttention(num_heads=num_heads, dim_in=dim_model, dim_out=dim_in),
            dim_out=dim_model,
            dropout=dropout
        )

        self._feed_forward = AddNormLayer(
            feed_forward(dim_model, dim_feedforward),
            dim_out=dim_model,
            dropout=dropout
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._feed_forward(self._attention(X, X, X))
        
class TransformerEncoder(torch.nn.Module):
    def __init__(self, num_layers: int = 6, dim_model: int = 512, num_heads: int = 6, dropout: float = 0.1, dim_feedforward: int = 2048) -> None:
        super().__init__()
        self._layers = [TransformerEncoderLayer(dim_model=dim_model, num_heads=num_heads, dropout=dropout, dim_feedforward=dim_feedforward) for _ in range(num_layers)]
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        seq_len, num_features = X.size(1), X.size(2)
        pos = positional_encoding(sequence_length=seq_len, dim_in=num_features)
        X += pos
        for layer in self._layers:
            X = layer(X)
        return X

class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, dim_model: int = 512, num_heads: int = 6, dropout: float = 0.1, dim_feedforward: int = 2048) -> None:
        super().__init__()
        dim_in = max(dim_model // num_heads, 1)
        self._attention1 = AddNormLayer(
            MultiHeadAttention(num_heads=num_heads, dim_in = dim_model, dim_out=dim_in),
            dim_out=dim_model,
            dropout=dropout
        )

        self._attention2 = AddNormLayer(
            MultiHeadAttention(num_heads=num_heads, dim_in=dim_model, dim_out=dim_in),
            dim_out=dim_model,
            dropout=dropout
        )

        self._feed_forward = AddNormLayer(
            feed_forward(dim_model, dim_feedforward),
            dim_out=dim_model,
            dropout=dropout
        )

    def forward(self, target: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        target = self._attention1(target, target, target)
        target = self._attention2(target, memory, memory)
        return self._feed_forward(target)

class TransformerDecoder(torch.nn.Module):
    def __init__(self, num_layers: int = 6, dim_model: int = 512, num_heads: int = 6, dropout: float = 0.1, dim_feedforward: int = 2048) -> None:
        super().__init__()
        self._layers = [
            TransformerDecoderLayer(
                dim_model=dim_model,
                num_heads=num_heads,
                dropout=dropout, 
                dim_feedforward=dim_feedforward
            ) for _ in range(num_layers)
        ]
        self._linear = torch.nn.Linear(dim_model, dim_model)

    def forward(self, target: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        seq_len, num_features = target.size(1), target.size(2)
        target += positional_encoding(sequence_length=seq_len, dim_in=num_features)
        for layer in self._layers:
            target = layer(target, memory)

        return torch.softmax(self._linear(target), dim=-1)

class Transformer(torch.nn.Module):
    def __init__(self,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_model: int = 512, 
        num_heads: int = 6, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        activation: torch.nn.Module = torch.nn.ReLU()) -> None:
        super().__init__()
        self._encoder = TransformerEncoder(
            num_layers=num_encoder_layers, 
            dim_model=dim_model, 
            num_heads=num_heads, 
            dropout=dropout,
            dim_feedforward=dim_feedforward)

        self._decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward
        )
        self.CLS_TOKEN = torch.nn.Parameter(torch.rand((dim_model,), requires_grad=False))

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._decoder(target, self._encoder(source))


class AudioSpectrogramTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.nn.Conv2d(1, )

    def forward(self, X):
        """X must have shape (batch_size, 1, frequency_bins, n_time_steps)"""
        batch_size = X.shape[0]
        return self.forward(X)


if __name__ == "__main__":
    batch_size = 42
    num_features = 128
    in_sequence_length = 200
    out_sequence_length = 10
    input_shape = (batch_size, in_sequence_length, num_features)
    output_shape = (batch_size, out_sequence_length, num_features)
    
    # in_sequence_length += 1 # for CLS token
    
    raw_src = torch.rand(input_shape)
    CLS_TOKEN = torch.rand((num_features,)) # shape [128]
    src = prepend_cls_token(raw_src, CLS_TOKEN)

    tgt = torch.rand(output_shape)

    # model = Transformer(
    #     num_encoder_layers=6,
    #     num_decoder_layers=6,
    #     dim_model=num_features,
    #     num_heads=6,
    #     dim_feedforward=2048,
    #     dropout=0.1,
    #     activation=torch.nn.ReLU()
    # )
    model = TransformerEncoder(
        num_layers=6,
        dim_model=num_features,
        num_heads=6,
        dropout=0.1,
        dim_feedforward=2048
    )
    out = model(src)
    print(out.shape)

    