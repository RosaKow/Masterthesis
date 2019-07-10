#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import matplotlib.pyplot as plt
import numpy as np
import pickle

n = 200
def v(lxy,sigv):
        out = np.zeros_like(lxy)
        mom = np.random.normal(0,1,np.shape(lxy))
        print(np.shape(lxy))
        for i in range(np.shape(lxy)[0]):
            out[i,:] = np.dot(np.exp(-((lxy[:,0]-lxy[i,0])**2+(lxy[:,1]-lxy[i,1])**2)/(2*sigv**2)),mom)
        return out

def fonc(p,w,sig):
        def f(u,v):
            values = np.zeros_like(u);
            for i in range(2):
                values += w[i]*np.exp(-((u-p[i][0])**2+(v-p[i][1])**2)/(2*sig[i]**2))
            return values
        return f    
lines = []

for k in range(15):
    lim = 5.
    x, y = np.mgrid[-lim:lim:n*1j, -lim:lim:n*1j]
    p = np.array([[-1, 0.], [1., 0]]) 
    shift = np.random.normal(0,1,(2,))
    
    sig = np.array([0.6, 0.6])
    sigv = 0.2
    w = np.random.rand(2) + 0.2
    w = w/w.sum()    
    f = fonc(p,w,sig)
    a = np.linspace(0,1,20)
    zx = p[0][0]*a+(1-a)*p[1][0]
    zy = p[0][1]*a+(1-a)*p[1][1]
    
    thresh = 0.95*np.min(f(zx,zy))
    fig1 = plt.figure(1)
    cs = plt.contour(x, y, f(x-shift[0],y-shift[1]), levels = [thresh])
    
    
    for line in cs.collections[0].get_paths():
        lines.append(line.vertices + 0.01*v(line.vertices,sigv))
    
    fig2 = plt.figure(2)
    ax2 = plt.gca()
    
    ax2.plot(lines[k][:, 0], lines[k][:, 1])
    plt.axis('equal')
plt.show()
plt.savefig("nutsdata.pdf", dpi =300)

with open('nutsdata.pickle', 'wb') as my_f:
    pickle.dump([lines, sigv, sig], my_f)