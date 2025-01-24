import torch
import torch.nn as nn
from torch.nn.functional import softmax
import math

# one head of attention, with optional padding and causal masking for use in any part of the transformer
class Head(nn.Module):

    def __init__(self, head_dim, n_embedding, device):
        super().__init__()
        # setup keys, queries, values
        self.head_dim = head_dim # will be useful to scale dot products
        self.key = nn.Linear(n_embedding, head_dim, bias=False)
        self.query = nn.Linear(n_embedding, head_dim, bias=False)
        self.value = nn.Linear(n_embedding, head_dim, bias=False)
        # optional: add dropout
        self.dropout = nn.Dropout(0.1)
        self.device = device

    def forward(self, x, k=None, v=None, padding_mask=None, add_causal_mask=False): # allow to specify k and v if doing cross attention
        # padding_mask: B, T before unsqueezing
        B, T, C = x.shape
        if k is None:
            k = self.key(x) # B, T, head_dim
        q = self.query(x)
        if v is None:
            v = self.value(x) # B, T, head_dim
        # (KQT) / sqrt(dim_head)
        attentions = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # B, T, head_dim @ B, head_dim, T -> B, T, T
        if padding_mask is not None:
            attentions = attentions.masked_fill(padding_mask.unsqueeze(1) == 0, float('-inf')) # B, T, T
        if add_causal_mask:
            attentions = attentions.masked_fill(torch.tril(torch.ones(T, T, device=self.device)) == 0, float('-inf')) # B, T, T
        attentions = softmax(attentions, dim=-1)
        attentions = self.dropout(attentions)
        return attentions @ v

class MultiHeadAttention(nn.Module):

    # num_heads * head_dimension == n_embedding
    def __init__(self, num_heads, head_dim, n_embedding, device):
        super().__init__()

        self.head_dim = head_dim
        self.heads = nn.ModuleList([Head(self.head_dim, n_embedding, device) for _ in range(num_heads)])
        self.project = nn.Linear(n_embedding, n_embedding) # to project after concatenation
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, enc_out=None, padding_mask=None, add_causal_mask=False):
        if enc_out is None:
            res = torch.cat([head.forward(x, None, None, padding_mask, add_causal_mask) for head in self.heads], dim=-1)
        else:
            k_v = [enc_out[:,:,i:i+self.head_dim] for i in range(x.shape[-1])]
            res = torch.cat([head.forward(x, kv, kv, padding_mask, add_causal_mask) for head, kv in zip(self.heads, k_v)], dim=-1)

        res = self.project(res)
        res = self.dropout(res)
        return res

# simple MLP with one hidden dim for use in both encoder and decoder blocks
class FeedForward(nn.Module):

    def __init__(self, n_embedding):
        super().__init__()
        self.network = nn.Sequential(
            # n_embedding -> 4 * n_embedding -> n_embedding
            nn.Linear(n_embedding, 4 * n_embedding),
            nn.ReLU(),
            nn.Linear(4 * n_embedding, n_embedding),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.network.forward(x)

class EncoderBlock(nn.Module):

    def __init__(self, n_embedding, n_head, device):
        super().__init__()
        head_dim = n_embedding // n_head
        self.self_attention = MultiHeadAttention(n_head, head_dim, n_embedding, device)
        self.ff = FeedForward(n_embedding)
        self.norm1 = nn.LayerNorm(n_embedding)
        self.norm2 = nn.LayerNorm(n_embedding)

    def forward(self, x, padding_mask):
        x = self.norm1(self.self_attention.forward(x, padding_mask=padding_mask) + x)
        x = self.norm2(self.ff.forward(x) + x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, n_embedding, n_head, device):
        super().__init__()
        head_dim = n_embedding // n_head
        self.self_attention = MultiHeadAttention(n_head, head_dim, n_embedding, device)
        self.cross_attention = MultiHeadAttention(n_head, head_dim, n_embedding, device)
        self.ff = FeedForward(n_embedding)
        self.norm1 = nn.LayerNorm(n_embedding)
        self.norm2 = nn.LayerNorm(n_embedding)
        self.norm3 = nn.LayerNorm(n_embedding)

    def forward(self, x, enc_out, padding_mask): # k, v supplied by encoder
        x = self.norm1(self.self_attention.forward(x, padding_mask=padding_mask, add_causal_mask=True) + x)
        x = self.norm2(self.cross_attention.forward(x, enc_out, padding_mask=padding_mask, add_causal_mask=True) + x)
        x = self.norm3(self.ff.forward(x) + x)
        return x

class Transformer(nn.Module):

    def __init__(self, n_layers, n_embedding, vocab_size, block_size, device, pad_token, start_token, end_token):
        super().__init__()
        self.enc_token_embeddings = nn.Embedding(vocab_size, n_embedding)
        self.dec_token_embeddings = nn.Embedding(vocab_size, n_embedding)
        self.position_embeddings = nn.Embedding(block_size, n_embedding)
        self.encoder_blocks = nn.Sequential(*[EncoderBlock(n_embedding, 16, device) for _ in range(n_layers)])
        self.decoder_blocks = nn.Sequential(*[DecoderBlock(n_embedding, 16, device) for _ in range(n_layers)])
        self.final_linear = nn.Linear(n_embedding, vocab_size)
        self.device = device
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token

        # softmax after to generate probabilities

    def forward(self, src, src_padding_mask, targets=None, generated=None):
        # === Encoder ===
        B, T = src.shape
        # pass input into encoder
        enc_embed = self.enc_token_embeddings(src) + self.position_embeddings(torch.arange(T, device=self.device))
        for block in self.encoder_blocks:
            enc_embed = block.forward(enc_embed, src_padding_mask)
        enc_out = enc_embed
        # === Decoder ===
        if generated is not None:
            dec_embed = self.dec_token_embeddings(generated) + self.position_embeddings(torch.arange(T, device=self.device))
        elif targets is not None:
            # add start of sentence token to each, concatenate column-wise to targets with final tokens for each removed
            # still will have block_size cols
            dec_embed = (self.dec_token_embeddings(torch.cat((torch.full((targets.shape[0],1), fill_value=62509, device=self.device), targets[:, :-1]), dim=1))
                         + self.position_embeddings(torch.arange(T, device=self.device)))

        for block in self.decoder_blocks:
            dec_embed = block.forward(dec_embed, enc_out, padding_mask=src_padding_mask)

        dec_out = dec_embed
        logits = self.final_linear(dec_out)
        loss = None

        if targets is not None: # if training
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=62508)

        return logits, loss

    def generate(self, src, src_padding_mask, max_len=128): # src: english
        generated = torch.full(src.shape, fill_value=self.pad_token, device=self.device)
        generated[:,0] = self.start_token
        for t in range(1, max_len):
            logits, loss = self.forward(src, src_padding_mask, generated=generated)
            # focus only on latest token's logits
            last_logits = logits[:, t-1, :]

            probs = softmax(last_logits, dim=-1)

            # generate a concrete sample
            next = torch.multinomial(probs, num_samples=1)

            generated[:, t] = next.squeeze(1)

            # if all sentences have eos, stop
            if torch.all(next == self.end_token):
                break

        return generated
