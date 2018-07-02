#Reference: E. Burman, M. Nechita, L. Oksanen, Unique continuation for the Helmholtz equation using stabilized finite element methods, J. Math. Pures Appl., 2018
#computing the errors and the value of the stabilizers for an approximate solution
from dolfin import *

#computing the H^1_0 error on a subdomain
def error_normH10(ue, u, ind_B, mesh, degree):
    locW = FunctionSpace(mesh, 'P', degree + 3)
    ue_W = interpolate(ue, locW)
    u_W = interpolate(u, locW)
    e_W = Function(locW)
    e_W.vector()[:] = ue_W.vector().array() - u_W.vector().array()
    error = ind_B * dot(grad(e_W), grad(e_W)) * dx
    return sqrt(abs(assemble(error)))

def error_list(ue, u, z, ind_B):
    fspace = u.function_space()    
    mesh = fspace.mesh()
    degree = fspace.ufl_element().degree()
    
    #errors in B
    err_B_L2 = sqrt(assemble(ind_B * dot(ue-u, ue-u) * dx)) / sqrt(assemble(ind_B * dot(ue, ue) *dx(mesh)))
    err_B_H10 = error_normH10(ue, u, ind_B, mesh, degree) / error_normH10(ue, Constant(0), ind_B, mesh, degree)
    
    #primal stabilizer
    pgamma = 1e-5
    n = FacetNormal(mesh)
    prs = assemble(pgamma * dot(jump(grad(u),n), jump(grad(u),n)) * dS + pgamma * jump(div(grad(u))) * jump(div(grad(u))) * dS)
    
    #dual stabilizer
    dus = sqrt(assemble(dot(grad(z), grad(z))*dx))
    r = [fspace.dim(), err_B_L2, err_B_H10, prs, dus]
    return r
