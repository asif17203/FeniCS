import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 1e-2  # Regularization parameter
tol = 1e-4  # Tolerance for convergence
max_iter = 100  # Maximum number of iterations

# Define the mesh and function space
mesh = fe.UnitSquareMesh(32, 32)
V = fe.FunctionSpace(mesh, 'P', 1)

# Define the desired state zd (using a desired state function)
zd_expr = fe.Expression("x[0]*x[1]", degree=2)
zd = fe.interpolate(zd_expr, V)

# Define initial control (zero function)
u = fe.Function(V)
u_prev = fe.Function(V)
u.assign(fe.interpolate(fe.Constant(0), V))

# Define test functions
v = fe.TestFunction(V)

# Define the variational problem for the state equation
y = fe.Function(V)
y_trial = fe.TrialFunction(V)
a_state = (fe.dot(fe.grad(y_trial), fe.grad(v)) + y_trial**3*v) * fe.dx
L_state = u * v * fe.dx

# Define the variational problem for the adjoint equation
p = fe.Function(V)
p_trial = fe.TrialFunction(V)
a_adjoint = (fe.dot(fe.grad(p_trial), fe.grad(v)) + 3*y**2*p_trial*v) * fe.dx
L_adjoint = (y - zd) * v * fe.dx

# Gradient of the cost functional
def compute_gradient(y, p, u):
    return fe.project(-p - alpha * u, V)

# Armijo line search
def armijo_line_search(u, y, p, grad, alpha_init=1, rho=0.5, c=1e-4):
    beta = alpha_init
    u_trial = fe.Function(V)
    while True:
        u_trial.assign(u + beta * grad)
        y.assign(solve_state(u_trial))
        cost1 = 0.5 * fe.assemble((y - zd)**2 * fe.dx) + alpha / 2 * fe.assemble(u**2 * fe.dx)
        cost2 = 0.5 * fe.assemble((y - zd)**2 * fe.dx) + alpha / 2 * fe.assemble(u_trial**2 * fe.dx)
        if cost2 <= cost1 + c * beta * fe.assemble(fe.dot(grad, grad) * fe.dx):
            break
        beta *= rho
    return beta

# Solve the state equation
def solve_state(u):
    fe.solve(a_state == L_state, y)
    return y

# Solve the adjoint equation
def solve_adjoint(y):
    fe.solve(a_adjoint == L_adjoint, p)
    return p

# Optimization loop
for k in range(max_iter):
    # Solve state equation
    y.assign(solve_state(u))
    
    # Solve adjoint equation
    p.assign(solve_adjoint(y))
    
    # Compute gradient
    grad = compute_gradient(y, p, u)
    
    # Line search
    alpha_k = armijo_line_search(u, y, p, grad)
    
    # Update control
    u_prev.assign(u)
    u.assign(u + alpha_k * grad)
    
    # Check convergence
    if fe.errornorm(u, u_prev) < tol:
        print(f"Converged after {k+1} iterations.")
        break

# Extract data for Matplotlib plotting
u_values = u.compute_vertex_values(mesh)
x, y = mesh.coordinates().T

# Plotting using Matplotlib
plt.figure(figsize=(8, 6))
plt.tricontourf(x, y, u_values, levels=50, cmap='viridis')
plt.colorbar(label='Control Function u')
plt.title('Optimal Control Function')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



