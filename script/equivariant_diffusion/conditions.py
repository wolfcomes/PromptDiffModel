import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.view(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class CategoricalEmbedder(nn.Module):
    """
    Embeds categorical conditions such as data sources into vector representations. 
    Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None, t=None):
        labels = labels.long().view(-1)
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        if True and train:
            noise = torch.randn_like(embeddings)
            embeddings = embeddings + noise
        return embeddings
    
class ClusterContinuousEmbedder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0

        if use_cfg_embedding:
            self.embedding_drop = nn.Embedding(1, hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.Softmax(dim=1),
            nn.Linear(hidden_size, hidden_size, bias=False)
        )
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

    def forward(self, labels, train, force_drop_ids=None, timestep=None):
        use_dropout = self.dropout_prob > 0
        if force_drop_ids is not None:
            drop_ids = force_drop_ids == 1
        else:
            drop_ids = None

        if (train and use_dropout):
            drop_ids_rand = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            if force_drop_ids is not None:
                drop_ids = torch.logical_or(drop_ids, drop_ids_rand)
            else:
                drop_ids = drop_ids_rand
        
        if drop_ids is not None:
            embeddings = torch.zeros((labels.shape[0], self.hidden_size), device=labels.device)
            embeddings[~drop_ids] = self.mlp(labels[~drop_ids])
            embeddings[drop_ids] += self.embedding_drop.weight[0]
        else:
            embeddings = self.mlp(labels)

        if train:
            noise = torch.randn_like(embeddings)
            embeddings = embeddings + noise
        return embeddings
