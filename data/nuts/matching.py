
import pickle
import time

import torch

import matplotlib.pyplot as plt
import implicitmodules.torch as im
import implicitmodules.torch.DeformationModules.ConstrainedTranslations as constr_trans
import implicitmodules.torch.Attachment.attachement as attach

#%%
dty = torch.float32
source = torch.tensor([[-1., -1.], [1., 1.], [1., -1.], [-1., 1.]], requires_grad=True, dtype=dty)
#target = torch.tensor([[0., 0.], [2., 0.], [1., -1.], [1., 1.]], requires_grad=True, dtype=dty)

target = torch.tensor([[-2., -2.], [2., 2.], [2., -2.], [-2., 2.]], requires_grad=True, dtype=dty)
#target = 1.+ torch.tensor([[-1., -1.], [1., 1.], [1., -1.], [-1., 1.]], requires_grad=True, dtype=dty)
def close_loop(oc):
    cc = np.zeros((np.shape(oc)[0]+1,2))
    cc[0:-1,:] = oc
    cc[-1][:] = oc[0][:]
    return cc
#%%


with open('implicitmodules/data/nutsdata.pickle', 'rb') as f:
    lines, sigv, sig = pickle.load(f)
source = torch.tensor(lines[0][::2], requires_grad=True, dtype=dty)[1:]

target = torch.tensor(lines[1][::2]  , requires_grad=True, dtype=dty)[1:]
num_target = 1

target = torch.tensor(lines[7][::2]  , requires_grad=True, dtype=dty)[1:]
num_target = 7



pts_source = source.detach().numpy()
#%%


sigma_scaling = 1.
a = torch.sqrt(torch.tensor(3.))
direc_scaling_pts = torch.tensor([[1., 0.], [-0.5 , 0.5* a],  [-0.5, -0.5* a]], requires_grad=True, dtype=dty)
direc_scaling_vec =  torch.tensor([[1., 0.], [-0.5 , 0.5* a],  [-0.5, -0.5* a]], requires_grad=True, dtype=dty)
def f(x):
    centre = x.view(1,2).repeat(3,1)
    return centre + 0.3 * sigma_scaling * direc_scaling_pts

def g(x):
    return direc_scaling_vec

#%%
#gd0 = torch.tensor([[-1., 0.6]], requires_grad=True, dtype=dty)
gd0 = torch.tensor([[-1., 0.]], requires_grad=True, dtype=dty)
cotan0 = torch.tensor([[0., 0.]], requires_grad=True, dtype=dty)
#gd1 = torch.tensor([[0.8, 0.6]], requires_grad=True, dtype=dty)
gd1 = torch.tensor([[1., 0.]], requires_grad=True, dtype=dty)
cotan1 = torch.tensor([[0., 0.]], requires_grad=True, dtype=dty)
#%%
pts = f(gd0).detach().numpy()
vec = g(gd0).detach().numpy()
pts1 = f(gd1).detach().numpy()
vec1 = g(gd1).detach().numpy()


plt.quiver(pts[:,0], pts[:,1], vec[:,0], vec[:,1])
plt.quiver(pts1[:,0], pts1[:,1], vec1[:,0], vec1[:,1])
plt.plot(source.detach().numpy()[:,0], source.detach().numpy()[:,1], 'b')
plt.plot(target.detach().numpy()[:,0], target.detach().numpy()[:,1], 'r')
plt.axis('equal')
#%%
scaling0 = constr_trans.ConstrainedTranslations(im.Manifolds.Landmarks(2, 1, gd = gd0.view(-1), cotan = cotan0.view(-1)), f, g, sigma_scaling)
scaling1 = constr_trans.ConstrainedTranslations(im.Manifolds.Landmarks(2, 1, gd = gd1.view(-1), cotan = cotan1.view(-1)), f, g, sigma_scaling)


sigma00 = 400.
nu00 = 0.001
coeff00 = 10.
implicit00 = im.DeformationModules.ImplicitModule0(
    im.Manifolds.Landmarks(2, 1, gd=torch.tensor([0., 0.], requires_grad=True)), sigma00, nu00, coeff00)


sigma0 = 0.2
nu0 = 0.001
coeff0 = 10.
implicit0 = im.DeformationModules.ImplicitModule0(
    im.Manifolds.Landmarks(2, pts_source.shape[0], gd=torch.tensor(pts_source, requires_grad=True).view(-1)), sigma0, nu0, coeff0)


sigma1 = 0.2
nu1 = 0.001
coeff1 = 10.
implicit1 = im.DeformationModules.ImplicitModule0(
    im.Manifolds.Landmarks(2, pts_source.shape[0], gd=torch.tensor(pts_source, requires_grad=True).view(-1)), sigma1, nu1, coeff1)



#%%
#model = im.Models.ModelPointsRegistration(
#    [source],
#    [scaling0, scaling1, implicit00, implicit0],
#    [True, True, True, True],
#    [attach.VarifoldAttachement([1, 0.2])]
#)
#type_exp = '_parametric_'  + str(num_target)
# 


model = im.Models.ModelPointsRegistration(
    [source],
    [implicit00, implicit1],
    [True, True],
    [attach.VarifoldAttachement([1, 0.2])]
)
type_exp = '_lddmm_'  + str(num_target)


#%%
fitter = im.Models.ModelFittingScipy(model, 1., 100.)
#%%

costs = fitter.fit([target], 750, options={})
#costs = model.fit([(target, torch.ones(target.shape[0], requires_grad=True))], max_iter=2000, l=100., lr=0.0001, log_interval=1)

#%%
model.compute([target])
import numpy as np
grid_origin, grid_size, grid_resolution = [-2., -1.], [4., 2.], [50,25]
def_grids = model.compute_deformation_grid(grid_origin, grid_size, grid_resolution, it=10)

final = model.modules[0].manifold.gd.view(-1,2).detach().numpy()
ax_c = plt.subplot(111, aspect='equal')
im.Utilities.usefulfunctions.plot_grid(ax_c, def_grids[-1][0].numpy(), def_grids[-1][1].numpy(), color='k')

plt.plot(source.detach().numpy()[:,0], source.detach().numpy()[:,1], 'b', label='source')
plt.plot(target.detach().numpy()[:,0], target.detach().numpy()[:,1], 'r', label='target')
plt.plot(final[:,0], final[:,1], 'G', label='final')
GD0 = model.init_manifold[1].gd.detach().numpy()
GD1 = model.init_manifold[2].gd.detach().numpy()
GD_opt = np.concatenate([GD0, GD1]).reshape([-1, 2])
plt.plot(GD_opt[:,0], GD_opt[:,1], '+b', label='optimized')
names_exp = 'exp0'
gd_init = np.array([[-1., 0.6], [0.8, 0.6]])
#names_exp = 'exp_LDDMM_grande_translation_sigma01'
gd_init = np.array([[-1., 0.], [1., 0.]])
plt.plot(gd_init[:,0], gd_init[:,1], '*b', label='initialisation')
plt.axis('equal')
plt.axis([-3, 3,-3,3])
plt.legend()
path = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/Results/DeformationModule/cacahuete/'
plt.savefig(path + names_exp)
#%% 
grid_origin, grid_size, grid_resolution = [-2., -1.], [4., 2.], [20, 10]
def_grids = model.compute_deformation_grid(grid_origin, grid_size, grid_resolution, it=10)

ax_c = plt.subplot(111, aspect='equal')
plt.axis('equal')
plt.title('Source')
plt.xlabel('x')
plt.ylabel('y')
im.Utilities.usefulfunctions.plot_grid(ax_c, def_grids[-1][0].numpy(), def_grids[-1][1].numpy(), color='k')

plt.plot(source.detach().numpy()[:,0], source.detach().numpy()[:,1], 'b', label='source')
plt.plot(target.detach().numpy()[:,0], target.detach().numpy()[:,1], 'r', label='target')
plt.plot(final[:,0], final[:,1], 'k', label='final')
GD0 = model.init_manifold[1].gd.detach().numpy()
GD1 = model.init_manifold[2].gd.detach().numpy()
GD_opt = np.concatenate([GD0, GD1]).reshape([-1, 2])
plt.plot(GD_opt[:,0], GD_opt[:,1], '+b', label='optimized')


#%%
source_shot = model.compute([(target, torch.ones(target.shape[0], dtype=dty))])

#%%
import pickle
path = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/Results/DeformationModule/cacahuete/'

name = 'exp_nuts_matching' + type_exp + 'modelsaved.pickle'
pickle.dump(model, open(path + name, 'wb'))

#%%
from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.DeformationModules import CompoundModule, SilentLandmarks

#%%
from implicitmodules.torch.HamiltonianDynamic import shooting

from implicitmodules.torch.Utilities.usefulfunctions import grid2vec, vec2grid
from implicitmodules.torch.Manifolds import Landmarks

import numpy as np
#%%
grid_origin, grid_size, grid_resolution = [-2., -1.], [4., 2.], [50, 25]
frame = [-2, 3,  -2,1]
#%%

x, y = torch.meshgrid([
    torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
    torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])
gridpos = grid2vec(x, y)

grid_landmarks = Landmarks(2, gridpos.shape[0], gd=gridpos.view(-1))
grid_silent = SilentLandmarks(grid_landmarks)
#compound = CompoundModule(model.modules)
#compound.manifold.fill(model.init_manifold)


#intermediate_states= shoot(Hamiltonian([grid_silent, *compound]), 10, 'euler')

##%%

compound = CompoundModule(model.modules)
compound.manifold.fill(model.init_manifold)
h = Hamiltonian(compound)
intermediate_states, intermediate_controls = shoot(h, 10, "torch_euler")

##%%
compound = CompoundModule(model.modules)
compound.manifold.fill(model.init_manifold)
h = Hamiltonian(compound)
h = Hamiltonian([grid_silent, *compound])

controls_only_scaling = []
for t in range(10):
    c_t = []
    c_t.append(torch.tensor([], requires_grad=True))
    c_t.append(torch.tensor([], requires_grad=True))
    c_t.append(torch.tensor(intermediate_controls[t][1].clone().detach(), requires_grad=True))
    c_t.append(torch.tensor(intermediate_controls[t][2].clone().detach(), requires_grad=True))
    c_t.append(torch.zeros_like(intermediate_controls[t][3], requires_grad=True))
    c_t.append(torch.zeros_like(intermediate_controls[t][4], requires_grad=True))
    controls_only_scaling.append(c_t)

intermediate_states = shooting.shoot_euler_controls(h, controls_only_scaling, 10)
#intermediate_states.append(h.module.manifold)
##%%
k=0
name_init = 'matching' + type_exp
name_init += 'follow_controls_scaling'
curve_source = close_loop(source.detach().numpy())
curve_target = close_loop(target.detach().numpy())
for t in range(11):
    plt.figure()
    inter = intermediate_states[t].gd[0].view(-1, 2)
    gri = vec2grid(inter.detach(), grid_resolution[0], grid_resolution[1])
    ax_c = plt.subplot(111, aspect='equal')
    im.Utilities.usefulfunctions.plot_grid(ax_c, gri[0].numpy(), gri[1].numpy(), color='palevioletred')

    plt.plot(curve_source[:,0], curve_source[:,1], 'b', lw=3, label='source', alpha=0.6)
    #plt.plot(template.detach().numpy()[:,0], template.detach().numpy()[:,1], '-g', label='template optimized')
    plt.plot(curve_target[:,0], curve_target[:,1], 'k', label='target')
    
    cfinal = close_loop(intermediate_states[t].gd[1].view(-1, 2).detach().numpy())
    
    plt.plot(cfinal[:,0], cfinal[:,1], 'g', lw = 3, label='final', alpha=0.6)
    
    
    GD0 = model.init_manifold[1].gd.detach().numpy()
    GD1 = model.init_manifold[2].gd.detach().numpy()
    GD_opt = np.concatenate([GD0, GD1]).reshape([-1, 2])
    plt.plot(GD_opt[:,0], GD_opt[:,1], 'bo', label='optimized')
    
    
    GD0 = intermediate_states[t][2].gd.detach().numpy()
    GD1 = intermediate_states[t][3].gd.detach().numpy()
    GD_opt = np.concatenate([GD0, GD1]).reshape([-1, 2])
    plt.plot(GD_opt[:,0], GD_opt[:,1], 'go', label='final')
    
    plt.axis('equal')
    plt.axis(frame)
    plt.axis('off')
    name = name_init + 'grid_t_' + str(t)
    path = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/Results/DeformationModule/cacahuete/'
    plt.savefig(path + name, dpi=300, bbox_inches='tight')
    
plt.close('all')







#%%
#grid_origin, grid_size, grid_resolution = [-2., -1.], [4., 2.], [80, 40]

x, y = torch.meshgrid([
    torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
    torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])
gridpos = grid2vec(x, y)

grid_landmarks = Landmarks(2, gridpos.shape[0], gd=gridpos.view(-1))
grid_silent = SilentLandmarks(grid_landmarks)

compound = CompoundModule(model.modules)
compound.manifold.fill(model.init_manifold)
h = Hamiltonian(compound)
intermediate_states, intermediate_controls = shoot(h, 10, "torch_euler")

##%%
compound = CompoundModule(model.modules)
compound.manifold.fill(model.init_manifold)
h = Hamiltonian(compound)
h = Hamiltonian([grid_silent, *compound])
controls_without_scaling = []
for t in range(10):
    c_t = []
    c_t.append(torch.tensor([], requires_grad=True))
    c_t.append(torch.tensor([], requires_grad=True))
    c_t.append(torch.zeros_like(intermediate_controls[t][1], requires_grad=True))
    c_t.append(torch.zeros_like(intermediate_controls[t][2], requires_grad=True))
    c_t.append(torch.zeros_like(intermediate_controls[t][3], requires_grad=True))
    c_t.append(torch.tensor(intermediate_controls[t][4].clone().detach(), requires_grad=True))
    controls_without_scaling.append(c_t)

intermediate_states = shooting.shoot_euler_controls(h, controls_without_scaling, 10)
name_init = 'matching' + type_exp
name_init += 'follow_controls_small_trans'
curve_source = close_loop(source.detach().numpy())
curve_target = close_loop(target.detach().numpy())
for t in range(11):
    plt.figure()
    inter = intermediate_states[t].gd[0].view(-1, 2)
    gri = vec2grid(inter.detach(), grid_resolution[0], grid_resolution[1])
    ax_c = plt.subplot(111, aspect='equal')
    im.Utilities.usefulfunctions.plot_grid(ax_c, gri[0].numpy(), gri[1].numpy(), color='palevioletred')

    plt.plot(curve_source[:,0], curve_source[:,1], 'b', lw=3, label='source', alpha=0.6)
    #plt.plot(template.detach().numpy()[:,0], template.detach().numpy()[:,1], '-g', label='template optimized')
    plt.plot(curve_target[:,0], curve_target[:,1], 'k', label='target')
    
    cfinal = close_loop(intermediate_states[t].gd[1].view(-1, 2).detach().numpy())
    
    plt.plot(cfinal[:,0], cfinal[:,1], 'g', lw = 3, label='final', alpha=0.6)
    
    
    GD0 = model.init_manifold[1].gd.detach().numpy()
    GD1 = model.init_manifold[2].gd.detach().numpy()
    GD_opt = np.concatenate([GD0, GD1]).reshape([-1, 2])
    plt.plot(GD_opt[:,0], GD_opt[:,1], 'bo', label='optimized')
    
    
    GD0 = intermediate_states[t][2].gd.detach().numpy()
    GD1 = intermediate_states[t][3].gd.detach().numpy()
    GD_opt = np.concatenate([GD0, GD1]).reshape([-1, 2])
    plt.plot(GD_opt[:,0], GD_opt[:,1], 'go', label='final')
    
    plt.axis('equal')
    plt.axis(frame)
    plt.axis('off')
    name = name_init + 'grid_t_' + str(t)
    path = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/Results/DeformationModule/cacahuete/'
    plt.savefig(path + name, dpi=300, bbox_inches='tight')
    
plt.close('all')


#%%
#grid_origin, grid_size, grid_resolution = [-2., -1.], [4., 2.], [80, 40]

x, y = torch.meshgrid([
    torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
    torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])
gridpos = grid2vec(x, y)

grid_landmarks = Landmarks(2, gridpos.shape[0], gd=gridpos.view(-1))
grid_silent = SilentLandmarks(grid_landmarks)


compound = CompoundModule(model.modules)
compound.manifold.fill(model.init_manifold)
h = Hamiltonian(compound)
intermediate_states, intermediate_controls = shoot(h, 10, "torch_euler")

##%%
compound = CompoundModule(model.modules)
compound.manifold.fill(model.init_manifold)
h = Hamiltonian(compound)
h = Hamiltonian([grid_silent, *compound])
controls_large_trans = []
for t in range(10):
    c_t = []
    c_t.append(torch.tensor([], requires_grad=True))
    c_t.append(torch.tensor([], requires_grad=True))
    c_t.append(torch.zeros_like(intermediate_controls[t][1], requires_grad=True))
    c_t.append(torch.zeros_like(intermediate_controls[t][2], requires_grad=True))
    c_t.append(torch.tensor(intermediate_controls[t][3].clone().detach(), requires_grad=True))
    c_t.append(torch.zeros_like(intermediate_controls[t][4], requires_grad=True))
    controls_large_trans.append(c_t)

intermediate_states = shooting.shoot_euler_controls(h, controls_large_trans, 10)

name_init = 'matching' + type_exp
name_init += 'follow_controls_large_trans'
curve_source = close_loop(source.detach().numpy())
curve_target = close_loop(target.detach().numpy())
for t in range(11):
    plt.figure()
    inter = intermediate_states[t].gd[0].view(-1, 2)
    gri = vec2grid(inter.detach(), grid_resolution[0], grid_resolution[1])
    ax_c = plt.subplot(111, aspect='equal')
    im.Utilities.usefulfunctions.plot_grid(ax_c, gri[0].numpy(), gri[1].numpy(), color='palevioletred')

    plt.plot(curve_source[:,0], curve_source[:,1], 'b', lw=3, label='source', alpha=0.6)
    #plt.plot(template.detach().numpy()[:,0], template.detach().numpy()[:,1], '-g', label='template optimized')
    plt.plot(curve_target[:,0], curve_target[:,1], 'k', label='target')
    
    cfinal = close_loop(intermediate_states[t].gd[1].view(-1, 2).detach().numpy())
    
    plt.plot(cfinal[:,0], cfinal[:,1], 'g', lw = 3, label='final', alpha=0.6)
    
    
    GD0 = model.init_manifold[1].gd.detach().numpy()
    GD1 = model.init_manifold[2].gd.detach().numpy()
    GD_opt = np.concatenate([GD0, GD1]).reshape([-1, 2])
    plt.plot(GD_opt[:,0], GD_opt[:,1], 'bo', label='optimized')
    
    
    GD0 = intermediate_states[t][2].gd.detach().numpy()
    GD1 = intermediate_states[t][3].gd.detach().numpy()
    GD_opt = np.concatenate([GD0, GD1]).reshape([-1, 2])
    plt.plot(GD_opt[:,0], GD_opt[:,1], 'go', label='final')
    
    plt.axis('equal')
    plt.axis(frame)
    plt.axis('off')
    name = name_init + 'grid_t_' + str(t)
    path = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/Results/DeformationModule/cacahuete/'
    plt.savefig(path + name, dpi=300, bbox_inches='tight')
    
plt.close('all')

#%%
compound = CompoundModule(model.modules)
compound.manifold.fill(model.init_manifold)
h = Hamiltonian(compound)
controls_only_small_trans = []
for t in range(10):
    c_t = []
    c_t.append(torch.tensor([], requires_grad=True))
    c_t.append(torch.zeros_like(intermediate_controls[t][1], requires_grad=True))
    c_t.append(torch.zeros_like(intermediate_controls[t][2], requires_grad=True))
    c_t.append(torch.zeros_like(intermediate_controls[t][3], requires_grad=True))
    c_t.append(torch.tensor(intermediate_controls[t][4].clone().detach(), requires_grad=True))
    controls_only_small_trans.append(c_t)


#%%
from implicitmodules.torch.HamiltonianDynamic import shooting

intermediate_states = shooting.shoot_euler_controls(h, controls_no_small_trans, 10)

#intermediate_states.append(h.module.manifold)
#%%
k=0
name_init = 'matching_parametric_fixed_gd_'
name_init += 'follow_controls_no_small_trans'
for t in range(11):
    plt.figure()
    plt.plot(source.detach().numpy()[:,0], source.detach().numpy()[:,1], '-b', label='template initial')
    #plt.plot(template.detach().numpy()[:,0], template.detach().numpy()[:,1], '-g', label='template optimized')
    plt.plot(target.detach().numpy()[:,0], target.detach().numpy()[:,1], '-r', label='target', linewidth=1)
    
    curve = intermediate_states[t].gd[0].view(-1, 2).detach().numpy()
    plt.plot(curve[:,0], curve[:,1], 'g')
    plt.axis('equal')
    name = name_init + '_t_' + str(t)
    path = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/Results/DeformationModule/cacahuete/'
    plt.savefig(path + name)
    
plt.close('all')






