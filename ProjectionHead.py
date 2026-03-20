import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):  
        super().__init__()

        """
        Here you should write simple 2-layer MLP consisting:
        2 Linear layers, GELU activation, Dropout and LayerNorm. 
        Do not forget to send a skip-connection right after projection and before LayerNorm.
        The whole structure should be in the following order:
        [Linear, GELU, Linear, Dropout, Skip, LayerNorm]
        """
        
        self.linear1 = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        """
        Perform forward pass, do not forget about skip-connections.
        """
        x1 = self.linear1(x)
        x2 = self.gelu(x1)
        x3 = self.linear2(x2)
        x4 = self.dropout(x3)
        x_skip = x4 + x1
        x_out = self.layernorm(x_skip)
        return x_out