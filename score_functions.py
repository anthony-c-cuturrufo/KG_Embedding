#import torch.nn.functional as fn
import torch as th
import numpy as np
#head - shape(n, e)
#rel - shape(r) *r is usually set to e
#tail - shape(e)
#return - shape(n)

def transE_l2(head, rel, tail, gamma=12.0):
    score = head + rel - tail
    return gamma - th.norm(score, p=2, dim=-1) 

def transR(head,rel,tail,proj,rel_idx,gamma=12.0):   
    proj = proj.reshape(-1, head.shape[1], rel.shape[0])[rel_idx]
    head_r = th.einsum('ab,bc->ac', head, proj)
    tail_r = th.einsum('b,bc->c', th.tensor(tail), proj)
    score = head_r + rel - tail_r #25,40
    return gamma - th.norm(score, p=1, dim=-1) 

def complEx(head,rel,tail,gamma=12.0):
    real_head, img_head = th.chunk(head, 2, dim=-1)
    real_tail, img_tail = th.chunk(th.tensor(tail), 2, dim=-1)
    real_rel, img_rel = th.chunk(rel, 2, dim=-1)
    
    score = real_head * real_tail * real_rel \
            + img_head * img_tail * real_rel \
            + real_head * img_tail * img_rel \
            - img_head * real_tail * img_rel
    # TODO: check if there exists minus sign and if gamma should be used here(jin)
    return th.sum(score, -1)

def rotatE(head,rel,tail, emb_init=.5, gamma=12.0):
    #self.emb_init = [(gamma + self.eps) / hidden_dim] -> [12 + 2 / 40]
    re_head, im_head = th.chunk(head, 2, dim=-1)
    re_tail, im_tail = th.chunk(th.tensor(tail), 2, dim=-1)
    
    phase_rel = rel / (emb_init / np.pi)
    
    re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
    re_score = re_head * re_rel - im_head * im_rel
    im_score = re_head * im_rel + im_head * re_rel
    re_score = re_score - re_tail
    im_score = im_score - im_tail
    score = th.stack([re_score, im_score], dim=0)
    score = score.norm(dim=0)
    return gamma - score.sum(-1)