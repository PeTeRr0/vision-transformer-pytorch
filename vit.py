import torch
from torch import nn
import math

class PatchEmbedding(nn.Module):
  """
  Convert the image into patches and then project them into a vector space.
  """
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.img_size = config["img_size"]
    self.patch_size = config["patch_size"]
    self.channels = config["channels"]
    self.hidden_size = config["hidden_size"]
    self.num_patches = (self.img_size // self.patch_size) ** 2
    # Create a projection layer to convert the image into patches
    # The layer projects each patches into a vector (hidden size)
    self.projection = nn.Conv2d(
        self.num_patches,
        self.hidden_size,
        kernel_size=self.patch_size,
        stride=self.patch_size
    )
    # Creat a learnable [CLS] token
    # The [CLS] token is added to the beginning of the input sequence
    self.cls_token = nn.Parameter(torch.zeros(1, 1, config["self.hidden_size"]))
    # Create position embeddings for the [CLS] token and the patch embedidngs
    # Add 1 to the sequence length for the [CLS] token
    self.pos_embedding = nn.Parameter(torch.zeros(1, 1 + self.num_patches, config["self.hidden_size"]))
    self.dropout = nn.Dropout(config["dropout"])

  def forward(self, x):
    b, _, _, _ = x.shape
    x = self.projection(x)
    x = x.flatten(2)
    x = x.transpose(1, 2)

    # Expand the [CLS] token to the batch size
    # (1, 1, h) -> (b, 1, h)
    cls_tokens = self.cls_token.expand(b, -1, -1)
    # Concatenate the [CLS] token to the beginning of the input sequence
    # This results in a sequence length of (num_patches + 1)
    x = torch.cat((cls_tokens, x), dim=1)
    x += self.pos_embedding
    x = self.dropout(x)
    return x

class Attention(nn.Module):
  """
  Attention Head for MHA
  """
  def __init__(self, hidden_size, attention_head_size, dropout, bias=False):
    super().__init__()
    self.hidden_size = hidden_size
    self.attention_head_size = attention_head_size

    # Create the query, key and value projection layers
    self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
    self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
    self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # Project the input into query, key and value
    # The same input is used to generate the query, key and value
    # So it's called self-attention
    query = self.query(x)
    key = self.key(x)
    value = self.value(x)
    # Calculate attention score
    # softmax(q * k.T / sqrt(head_size)) * v
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
    attention_probs = torch.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)
    # Calculate the attention output
    attention_output = torch.matmul(attention_probs, value)
    return (attention_output, attention_probs)

class MultiHeadAttention(nn.Module):
  """
  MHA (Multi-Head Attention)
  """
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.hidden_size = config["hidden_size"]
    self.num_attention_heads = config["num_attention_heads"]
    self.attention_head_size = self.num_attention_heads * self.hidden_size
    # Whether or not to use bias in the query, key, and value porjection layers
    self.qkv_bias = config["qkv_bias"]
    # Create a list of attention heads
    self.heads = nn.ModuleList([])
    for _ in range(self.num_attention_heads):
      self.heads.append(Attention(self.hidden_size, self.attention_head_size, config["dropout"], self.qkv_bias))
    # Create a linear layer to project the attention output back to the hidden size
    self.output_projection = nn.Linear(self.attention_head_size, self.hidden_size)
    self.output_dropout = nn.Dropout(config["dropout"])

  def forward(self, x):
    # Calculate the attention output for each attention head
    attention_outputs = []
    for head in self.heads:
      attention_out, _ = head(x)
      attention_outputs.append(attention_out)
    # Concatenate the attention outputs from each attention head
    attention_output = []
    for attention_output in attention_outputs:
      attention_output = torch.cat(attention_outputs, dim=-1)
    # Project the concatenated attention output back to the hidden size
    attention_output = self.output_projection(attention_output)
    attention_output = self.output_dropout(attention_output)
    return attention_output

class MLP(nn.Module):
  """
  Multi-layer perceptron
  Feedforward
  """
  def __init__(self, config):
    super().__init__()
    self.net = nn.Sequential(
        nn.LayerNorm(config["hidden_size"]),
        nn.Linear(config["hidden_size"], config["intermediate_size"]),
        nn.GELU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config("intermediate_size"), config["hidden_size"]),
        nn.Dropout(config["dropout"])
    )

  def forward(self, x):
    return self.net(x)

class TransformerBlock(nn.Module):
  """
  Transformer Block
  """
  def __init__(self, config):
    super().__init__()
    nn.LayerNorm(config["hidden_size"])
    self.mlp = MLP(config)

  def forward(self, x):
    # Self-attention
    attention_output = MultiHeadAttention(x)
    x += attention_output
    # Feedforward
    mlp_output = self.mlp(x)
    x += mlp_output
    return x

class TransformerEncoder(nn.Module):
  """
  Transformer Encoder
  """
  def __init__(self, config):
    super().__init__()
    # Create a list of transformer blocks
    self.blocks = nn.ModuleList([])
    for _ in range(config["num_hidden_layers"]):
      self.blocks.append(TransformerBlock(config))

  def forward(self, x):
    # Calculate the transformer block's output for each block
    for block in self.blocks:
      x = block(x)

    return x

class ViT(nn.Module):
  """
  ViT model and classification
  """
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.img_size = config["img_size"]
    self.hidden_size = config["hidden_size"]
    self.num_classes = config["num_classes"]
    # Create the embedding module
    self.embedding = PatchEmbedding(config)
    # Create the transformer encoder module
    self.encoder = TransformerEncoder(config)
    # Create a linear layer to project the encoder's output to the number of classes
    self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])
    # Initialize the weights
    self.__init__weights()

  def forward(self, x):
    # Calculate the embedding output
    embedding_output = self.embedding(x)
    # Calculate the encoder's output
    encoder_output = self.encoder(embedding_output)
    # Calculate the logits take the [CLS] token's output as features for classification
    logits = self.classifier(encoder_output[:, 0])

    return logits

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
      module.weight.data.normal_(mean=0.0, std=0.02)
      if module.bias is not None:
        module.bias.data.zero_()