import torch
import math
pi = math.pi

def PointsOnCircle(origin, r,n):
    return [(origin[0] + math.cos(2*pi/n*x)*r,origin[1] + math.sin(2*pi/n*x)*r) for x in range(0,n)]

def multipleCircles(origin, r, n):
    circle1 = PointsOnCircle(origin[0], r[0], n[0])
    circle2 = PointsOnCircle(origin[1], r[1], n[1])
    return [torch.tensor(circle1, requires_grad=True), torch.tensor(circle2, requires_grad=True)] #[circle1, circle2]

def computeCenter(gd):
    return torch.mean(gd,0)

##############################################################

def shootMultishape(gd_list, mom_list, h, sigma, dim, n=10):
    step = 1. / n
    nb_points = [len(gd_list[0]), len(gd_list[1])]
    Constr1 = torch.cat([torch.eye(nb_points[0]), torch.zeros([nb_points[0], nb_points[1]]), -torch.eye(nb_points[0]), torch.zeros([nb_points[0], nb_points[1]])], 1)
    Constr2 = torch.cat([torch.zeros([nb_points[1], nb_points[0]]), torch.eye(nb_points[1]), torch.zeros([nb_points[1], nb_points[0]]), -torch.eye(nb_points[1]),], 1)
    Constr = [Constr1, Constr2]

    for i in range(n):  
        z = [computeCenter(gd_list[0]), computeCenter(gd_list[1])]
        
        
        l_list = lambda_qp(gd_list, mom_list, sigma, z, dim)
        controls_list = h_qp(gd_list, mom_list, l_list, sigma, z, dim)

        [d_gd1, d_gd2, d_gd3, d_mom1, d_mom2, d_mom3] = torch.autograd.grad(h(gd_list, mom_list, controls_list, z, l_list, Constr),[*gd_list, *mom_list], create_graph=True)
        
        gd_list[0] = gd_list[0] + step*d_mom1
        gd_list[1] = gd_list[1] + step*d_mom2
        gd_list[2] = gd_list[2] + step*d_mom3
        mom_list[0] = mom_list[0] - step*d_gd1
        mom_list[1] = mom_list[1] - step*d_gd2
        mom_list[2] = mom_list[2] - step*d_gd3
    return gd_list, mom_list


    
from kernels import K_xx, K_xy

def h_qp(gd, mom, l, sigma, z, dim):
    sigma1 = sigma[0]
    sigma2 = sigma[1]
    sigma3 = sigma[2]

    gd1 = gd[0].view(-1, dim).float()
    #z1 = torch.mm(torch.ones(len(gd1)).view(len(gd1),1),z[0].view(-1, dim).float())
    K1 = K_xy(z[0].view(-1, dim).float(), gd1, sigma1)
    
    hqp1 = torch.mm(K1,mom[0]) - torch.mm(K1,l[0])
    
    gd2 = gd[1].view(-1, dim).float()
    z2 = torch.mm(torch.ones(len(gd2)).view(len(gd2),1),z[1].view(-1, dim).float())
    K2 = K_xy(z[1].view(-1,dim).float(), gd2, sigma2)
    
    hqp2 = torch.mm(K2,mom[1]) - torch.mm(K2,l[1])
    
    gd3 = gd[2].view(-1, dim).float()
    K3 = K_xy(gd3, gd3, sigma3)
        
    hqp3 = torch.mm(K3, mom[2]) + torch.mm(K3, torch.cat([l[0], l[1]]))
    
    return [hqp1, hqp2, hqp3]


def lambda_qp(gd, mom, sigma, z, dim):
    
    sigma1 = sigma[0]
    sigma2 = sigma[1]
    sigma3 = sigma[2]
    
    nb_points = [len(gd[0]), len(gd[1])]
    
    mom31 = mom[2][0:len(gd[0])]
    mom32 = mom[2][len(gd[0]):len(mom[2])]

    z1 = z[0].view(-1, dim).float() #torch.tensor(origin[0]).view(-1, dim).float()
    gd1 = gd[0].view(-1, dim).float()
    K1 = K_xy(z1, gd1, sigma1)
    K31 = K_xx(gd1, sigma3) 
    
    K11 = torch.mm(torch.transpose(K1,0,1), K1)
    
    z2 = z[1].view(-1, dim).float() #torch.tensor(origin[1]).view(-1, dim).float()
    gd2 = gd[1].view(-1, dim).float()
    K2 = K_xy(z2, gd2, sigma2)
    K32 = K_xx(gd2, sigma3) 
    
    K22 = torch.mm(torch.transpose(K2,0,1), K2)
    
    #zeros = torch.zeros(nb_points[0], nb_points[1])
    #tmp1 = torch.cat([K11,  torch.transpose(zeros,0,1)],0)
    #tmp2 = torch.cat([zeros, K22],0)
    #K3 = torch.cat([torch.cat([K31,  torch.transpose(zeros,0,1)],0), torch.cat([zeros, K32],0)], 1)
    #K = torch.cat([tmp1, tmp2], 1)
    
    # X = C zeta Z^-1 zeta^ast C^ast
    X1 = K11 + K31
    X2 = K22 + K32
    
    # A = C zeta Z^-1 zeta^ast xi^ast
    A1 = torch.cat([K11, -K31], 1)
    A2 = torch.cat([K22, -K32], 1)
    
    lambdaqp1, _ = torch.gesv(torch.mm(A1,torch.cat([mom[0], mom31])),X1)
    lambdaqp2, _ = torch.gesv(torch.mm(A2,torch.cat([mom[1], mom32])),X2)

    
    return [lambdaqp1, lambdaqp2]