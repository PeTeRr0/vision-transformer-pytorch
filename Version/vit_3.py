
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from torch import nn
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, 
                 emb_size: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        
        # Initialize with proper scaling
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size) * 0.02)
        self.positions = nn.Parameter(torch.randn(self.n_patches + 1, emb_size) * 0.02)
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> qkv b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy = energy.masked_fill(~mask, fill_value)
            
        # Use proper scaling factor
        scaling = math.sqrt(self.emb_size)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        out = self.proj_drop(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.1):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
            )

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.1,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.1,
                 num_heads: int = 8):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size=emb_size, num_heads=num_heads),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 12, emb_size: int = 768, drop_p=0.1, 
                 forward_expansion=4, forward_drop_p=0.1, num_heads=8):
        super().__init__(*[
            TransformerEncoderBlock(
                emb_size=emb_size,
                drop_p=drop_p,
                forward_expansion=forward_expansion,
                forward_drop_p=forward_drop_p,
                num_heads=num_heads
            ) for _ in range(depth)
        ])
    
class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

class ViT(nn.Module):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                drop_p: float = 0.1,
                forward_expansion: int = 4,
                forward_drop_p: float = 0.1,
                num_heads: int = 8):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(
                depth=depth, emb_size=emb_size, drop_p=drop_p,
                forward_expansion=forward_expansion, forward_drop_p=forward_drop_p, num_heads=num_heads
            ),
            ClassificationHead(emb_size, n_classes)
        )