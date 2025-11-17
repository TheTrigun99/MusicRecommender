import torch
import torch.nn as nn
import torch.nn.functional as F
class Item2VecPaper(nn.Module):
    def __init__(self,num_items,emb_dim):
        super().__init__()
        self.num_items=num_items
        self.emb_dim=emb_dim
        self.emb_in=nn.Embedding(num_items, emb_dim)
        self.emb_out=nn.Embedding(num_items, emb_dim)
    
    def forward(self,center_ids, pos_ids, neg_ids):
        e_c= self.emb_in(center_ids) #embedding du centre (B, dim) B=batch size
        e_p=self.emb_out(pos_ids) 
        e_n=self.emb_out(neg_ids) #je crée embedding pur contexte (positif et négatif) (B, K, dim)
        cp= (e_c*e_p).sum(dim=1) #produit scalaire
        e_c_expanded = e_c.unsqueeze(1)   # dû aux dimensions différentes de e_n et e_c
        cn = (e_c_expanded * e_n).sum(dim=2)
        p_loss = F.logsigmoid(cp)
        n_loss = F.logsigmoid(-cn).sum(dim=1)  # (B,)

        loss = -(p_loss + n_loss).mean()
        return loss
    def get_embeddings(self):
        return self.emb_in.weight.detach()
    
    