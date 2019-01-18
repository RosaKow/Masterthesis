import torch
import numpy as np
"""from torch.autograd import grad"""

def shoot(Mod, GD, MOM, H_r, N):
    step = 1. / N
    GD_List = [GD]
    MOM_List = [MOM]
    for _ in range(N) :
        [d_GD, d_MOM] = torch.autograd.grad(H_r(GD, MOM), [GD, MOM], create_graph=True)
        GD = GD + step* d_MOM
        MOM = MOM - step* d_GD
        GD_List.append(GD)
        MOM_List.append(MOM)
    return GD_List, MOM_List