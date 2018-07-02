#Reference: E. Burman, M. Nechita, L. Oksanen, Unique continuation for the Helmholtz equation using stabilized finite element methods, J. Math. Pures Appl., 2018
#FEniCS implementation for the finited element method described in the above reference
from dolfin import *
import numpy as np

def boundary(x, on_boundary):
    return on_boundary

def stfem(ue, f, k, ind_omega, nele):
    k2 = Constant(k*k)
    mesh = UnitSquareMesh(nele, nele, 'right/left')
    hmax = mesh.hmax()
    h2 = hmax*hmax
    n = FacetNormal(mesh)

    V = FiniteElement("CG", mesh.ufl_cell(), 1)
    W  = FiniteElement("CG", mesh.ufl_cell(), 1)
    VW = FunctionSpace(mesh, V * W)
    
    (u, z) = TrialFunctions(VW) #discrete solution
    (v, w) = TestFunctions(VW)
    
    pgamma = Constant(1e-5) #fixed stabilization parameter
    a = ((dot(grad(u),grad(w)) - k2 * dot(u,w))) * dx \
      - dot(grad(z),grad(w)) * dx \
      + ind_omega * dot(u,v) * dx \
      + pgamma * hmax * dot(jump(grad(u),n), jump(grad(v),n)) * dS + pgamma * h2 * k2 * k2 * dot(u,v) * dx \
      + (dot(grad(v),grad(z)) - k2 * dot(v,z)) * dx
    L = f*w*dx + ind_omega*ue*v*dx
    
    w0 = Constant(0)
    boundary_cond = DirichletBC(VW.sub(1), w0, boundary) #impose zero Dirichlet boundary conditions on the space W
    
    sol = Function(VW)
    solve(a == L, sol, boundary_cond, solver_parameters={"linear_solver": "mumps"})
    (u, z) = sol.split()
    
    return [u, z]
