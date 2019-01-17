import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_momentum(GD, MOM, GD_final):
    "plots the momentum vectors and final GD points"
    
    GD_np = GD.detach().numpy().reshape([-1,2])
    MOM_np = MOM.detach().numpy().reshape([-1,2])
    GD_final_np = GD_final.detach().numpy().reshape([-1,2])
    
    plt.quiver(GD_np[:,0], GD_np[:,1], MOM_np[:,0], MOM_np[:,1], color=['r','b','g'], scale=21)
    plt.scatter(GD_final_np[:,0], GD_final_np[:,1], color=['r', 'b', 'g'], marker = 'x')
    plt.show()
    
def plot_vectorfield(Mod, GD, MOM):
    "plots the vectorfield resulting from GD and Mom"
    
    nx, ny = (10,10)
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    xv, yv = np.meshgrid(x, y)

    coordinate_grid = np.array([xv, yv])
    coordinate_grid = np.transpose(np.squeeze(np.reshape(coordinate_grid, (2,nx*ny,1))))

    Points = torch.tensor(coordinate_grid, dtype=torch.float32)
    
    Vectorfield = Mod(GD, MOM, Points).detach().numpy()

    plt.quiver(coordinate_grid[:,0], coordinate_grid[:,1], Vectorfield[:,0], Vectorfield[:,1], scale=21)
    plt.show()