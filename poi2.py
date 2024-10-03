import fenics as fe 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting toolkits
import numpy as np

N_points = 20
Forcing_Magnitude = 2.0

def main():
    mesh = fe.UnitSquareMesh(N_points, N_points)
    lagrange_polynomial_space_first_order = fe.FunctionSpace(
        mesh,
        "Lagrange",
        1,
    )
    def boundary_boolean_function(x, on_boundary):
        return on_boundary

    homogeneous_dirichlet_boundary_condition = fe.DirichletBC(
        lagrange_polynomial_space_first_order, 
        fe.Constant(0.0),
        boundary_boolean_function,
    )

    u_trial = fe.TrialFunction(lagrange_polynomial_space_first_order)
    v_test = fe.TestFunction(lagrange_polynomial_space_first_order)
    forcing = fe.Constant(-Forcing_Magnitude)
    weak_form_lhs = fe.dot(fe.grad(u_trial), fe.grad(v_test)) * fe.dx
    weak_form_rhs = forcing * v_test * fe.dx

    u_solution = fe.Function(lagrange_polynomial_space_first_order)
    fe.solve(
        weak_form_lhs == weak_form_rhs,
        u_solution,
        homogeneous_dirichlet_boundary_condition,
    )

    # Get coordinates and values of the solution
    mesh_coords = mesh.coordinates()
    u_values = u_solution.compute_vertex_values(mesh)
    
    # Prepare 3D plot using matplotlib
    x = mesh_coords[:, 0]
    y = mesh_coords[:, 1]
    z = u_values
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Triangulate the grid and plot the surface
    ax.plot_trisurf(x, y, z, cmap='hot')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    plt.show()

if __name__ == "__main__":
    main()
