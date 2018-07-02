#Reference: E. Burman, M. Nechita, L. Oksanen, Unique continuation for the Helmholtz equation using stabilized finite element methods, J. Math. Pures Appl., 2018
#definitions of computational domains
from dolfin import Expression
from numpy import pi

tol = 1e-14

#indicator for omega, Example 2, convex direction (24)
class ind_omega_conv(Expression):
    def set_values(self, i, o):
        self.i = i
        self.o = o
    def eval(self, value, x):
        if x[0]>(0.1-tol) and x[0]<(0.9+tol) and x[1]>(0.25-tol) and x[1]<(1+tol):
            value[0] = self.o
        else:
            value[0] = self.i
#indicator for B, Example 2, convex direction (24)            
class ind_B_conv(Expression):
    def set_values(self, i, o):
        self.i = i
        self.o = o
    def eval(self, value, x):
        if x[0]>(0.1-tol) and x[0]<(0.9+tol) and x[1]>(0.95-tol) and x[1]<(1+tol):
            value[0] = self.o
        else:
            value[0] = self.i
            
            
#indicator for omega, Example 2, non-convex direction (25)            
class ind_omega_nonconv(Expression):
    def set_values(self, i, o):
        self.i = i
        self.o = o
    def eval(self, value, x):
        if x[0]>(0.25-tol) and x[0]<(0.75+tol) and x[1]>(0-tol) and x[1]<(0.5+tol):
            value[0] = self.i
        else:
            value[0] = self.o
#indicator for B, Example 2, non-convex direction (25)
class ind_B_nonconv(Expression):
    def set_values(self, i, o):
        self.i = i
        self.o = o
    def eval(self, value, x):
        if x[0]>(0.125-tol) and x[0]<(0.875+tol) and x[1]>(0-tol) and x[1]<(0.95+tol):
            value[0] = self.i
        else:
            value[0] = self.o
