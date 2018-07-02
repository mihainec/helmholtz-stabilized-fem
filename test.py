#Reference: E. Burman, M. Nechita, L. Oksanen, Unique continuation for the Helmholtz equation using stabilized finite element methods, J. Math. Pures Appl., 2018
#checking the convergence of the method and plotting the approximate solution for ParaView
from dolfin import *
from stabilized_fem import *
from domains import *
from errors import *
import numpy as np
import matplotlib.pyplot as plt

k = 10

#indicator functions for omega and B, for one of the geometries defined in domains.py
ind_omega = ind_omega_conv(degree=0)
ind_omega.set_values(1, 0)
ind_B = ind_B_conv(degree=0)
ind_B.set_values(1, 0)

#gaussian bump in Example 2 from the paper, exact solution
ue = Expression("exp( -(x[0]-0.5)*(x[0]-0.5)/(2*sx) - (x[1]-1)*(x[1]-1)/(2*sy) )", degree=5, sx=0.01, sy=0.1)
#right hand side of the PDE
f = Expression("-( (x[0]-0.5)*(x[0]-0.5)/(sx*sx)-1/sx + (x[1]-1)*(x[1]-1)/(sy*sy)-1/sy + k*k ) * exp( -(x[0]-0.5)*(x[0]-0.5)/(2*sx) - (x[1]-1)*(x[1]-1)/(2*sy) )", degree=5, sx=0.01, sy=0.1, k=k)

#plot an approximate solution for visualization in ParaView
def plot_sol():
    u,z = stfem(ue, f, k, ind_omega, 128)
    vtkfile = File('conv_128_u.pvd')
    vtkfile << u

#check convergence (rates)
def convergence():
    nele = np.linspace(128,256,10)
    n = nele.size

    dof = np.zeros(n)
    h = np.zeros(n)    
    errL2 = np.zeros(n)
    errH10 = np.zeros(n)
    ps = np.zeros(n)
    ds = np.zeros(n)
    
    for i in range(n):
        u, z = stfem(ue, f, k, ind_omega, int(nele[i]))
        res = error_list(ue, u, z, ind_B)
        dof[i] = res[0]
        errL2[i] = res[1]
        errH10[i] = res[2]
        ps[i] = res[3]
        ds[i] = res[4]   
    h = 1/np.sqrt(dof)
    
    plt.figure()
    plt.loglog(h,errL2,'ks--',label='$L^2$-error',markerfacecolor='none')
    plt.loglog(h,errH10,'ko--',label='$H^1$-error',markerfacecolor='none')
    plt.loglog(h,ps,'kv--',label='$h^{-1}J(u_h,u_h)$',markerfacecolor='none')
    plt.loglog(h,ds,'k^--',label='$\|z_h\|_W$',markerfacecolor='none')
    plt.xlabel('log of meshsize')
    plt.ylabel('log')
    plt.grid(True)  
    plt.savefig('conv_rates.eps', format='eps')
    
    errs = [errL2, errH10, ps, ds]
    i=np.array(range(1,len(h)))
    #print estimated convergence rates
    for er in errs:
        print(np.log(er[i]/er[i-1])/np.log(h[i]/h[i-1]))
        print('\n')

plot_sol()
convergence()
