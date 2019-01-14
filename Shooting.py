import torch
import numpy as np
from torch.autograd import grad

def shoot(DefModule, GD, MOM, H_r, N):
    step = 1. / N
    for i in range(N) :
        [d_GD, d_MOM] = grad(H_r(GD, MOM), [GD, MOM], create_graph=True)
        GD = GD + step* d_MOM
        MOM = MOM - step* d_GD
    return GD, MOM

