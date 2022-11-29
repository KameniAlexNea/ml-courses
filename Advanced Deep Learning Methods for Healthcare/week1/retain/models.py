import torch
import torch.nn as nn


def attention_sum(alpha, beta, rev_v, rev_masks):
    """
    TODO: mask select the hidden states for true visits (not padding visits) and then
        sum the them up.

    Arguments:
        alpha: the alpha attention weights of shape (batch_size, seq_length, 1)
        beta: the beta attention weights of shape (batch_size, seq_length, hidden_dim)
        rev_v: the visit embeddings in reversed time of shape (batch_size, # visits, embedding_dim)
        rev_masks: the padding masks in reversed time of shape (# visits, batch_size, # diagnosis codes)

    Outputs:
        c: the context vector of shape (batch_size, hidden_dim)
        
    NOTE: Do NOT use for loop.
    """
    
    # your code here
    dot_val = ((alpha * beta) * rev_v) * rev_masks.max(dim=-1, keepdim=True).values
    return dot_val.sum(dim=-2)

def sum_embeddings_with_mask(x, masks):
    """
    Mask select the embeddings for true visits (not padding visits) and then sum the embeddings for each visit up.

    Arguments:
        x: the embeddings of diagnosis sequence of shape (batch_size, # visits, # diagnosis codes, embedding_dim)
        masks: the padding masks of shape (batch_size, # visits, # diagnosis codes)

    Outputs:
        sum_embeddings: the sum of embeddings of shape (batch_size, # visits, embedding_dim)
    """
    
    x = x * masks.unsqueeze(-1)
    x = torch.sum(x, dim = -2)
    return x


class AlphaAttention(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        """
        Define the linear layer `self.a_att` for alpha-attention using `nn.Linear()`;
        
        Arguments:
            hidden_dim: the hidden dimension
        """
        
        self.a_att = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        """
        TODO: Implement the alpha attention.
        
        Arguments:
            g: the output tensor from RNN-alpha of shape (batch_size, seq_length, hidden_dim) 
        
        Outputs:
            alpha: the corresponding attention weights of shape (batch_size, seq_length, 1)
            
        HINT: consider `torch.softmax`
        """
        
        # your code here
        proj = self.a_att(g)
        return torch.softmax(proj, dim=-2)

class BetaAttention(torch.nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        """
        Define the linear layer `self.b_att` for beta-attention using `nn.Linear()`;
        
        Arguments:
            hidden_dim: the hidden dimension
        """
        
        self.b_att = nn.Linear(hidden_dim, hidden_dim)


    def forward(self, h):
        """
        TODO: Implement the beta attention.
        
        Arguments:
            h: the output tensor from RNN-beta of shape (batch_size, seq_length, hidden_dim) 
        
        Outputs:
            beta: the corresponding attention weights of shape (batch_size, seq_length, hidden_dim)
            
        HINT: consider `torch.tanh`
        """
        
        # your code here
        proj = self.b_att(h)
        return torch.tanh(proj)

class RETAIN(nn.Module):
    
    def __init__(self, num_codes, embedding_dim=128):
        super().__init__()
        # Define the embedding layer using `nn.Embedding`. Set `embDimSize` to 128.
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        # Define the RNN-alpha using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_a = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        # Define the RNN-beta using `nn.GRU()`; Set `hidden_size` to 128. Set `batch_first` to True.
        self.rnn_b = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        # Define the alpha-attention using `AlphaAttention()`;
        self.att_a = AlphaAttention(embedding_dim)
        # Define the beta-attention using `BetaAttention()`;
        self.att_b = BetaAttention(embedding_dim)
        # Define the linear layers using `nn.Linear()`;
        self.fc = nn.Linear(embedding_dim, 1)
        # Define the final activation layer using `nn.Sigmoid().
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, masks, rev_x, rev_masks):
        """
        Arguments:
            rev_x: the diagnosis sequence in reversed time of shape (# visits, batch_size, # diagnosis codes)
            rev_masks: the padding masks in reversed time of shape (# visits, batch_size, # diagnosis codes)

        Outputs:
            probs: probabilities of shape (batch_size)
        """
        # 1. Pass the reversed sequence through the embedding layer;
        rev_x = self.embedding(rev_x)
        # 2. Sum the reversed embeddings for each diagnosis code up for a visit of a patient.
        rev_x = sum_embeddings_with_mask(rev_x, rev_masks)
        # 3. Pass the reversed embegginds through the RNN-alpha and RNN-beta layer separately;
        g, _ = self.rnn_a(rev_x)
        h, _ = self.rnn_b(rev_x)
        # 4. Obtain the alpha and beta attentions using `AlphaAttention()` and `BetaAttention()`;
        alpha = self.att_a(g)
        beta = self.att_b(h)
        # 5. Sum the attention up using `attention_sum()`;
        c = attention_sum(alpha, beta, rev_x, rev_masks)
        # 6. Pass the context vector through the linear and activation layers.
        logits = self.fc(c)
        probs = self.sigmoid(logits)
        return probs.squeeze()
    