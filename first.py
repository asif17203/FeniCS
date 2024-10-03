from fenics import  *
import matplotlib.pyplot as plt
import math as pi

if __name__ == "__main__":

 n_elements=32
 mesh = UnitIntervalMesh(n_elements)

 lagrange_polynomials_space_first_order=FunctionSpace(mesh , "Lagrange",1)

 u_on_boundary =Constant(0.0)

 def boundary_boolean_function(x ,on_bounrary):
     return on_bounrary

 boundary_condition =DirichletBC(lagrange_polynomials_space_first_order,u_on_boundary,boundary_boolean_function,)

 initial_condition =Expression ("sin(pi)*x[0]",degree=1)

 u_old=interpolate(initial_condition,lagrange_polynomials_space_first_order)
 
 time_step_length=0.1
 heat_source =Constant(0.0)

 u_trial =Constant(0.0)

 u_trial=TrialFunction(lagrange_polynomials_space_first_order)
 v_test =TestFunction(lagrange_polynomials_space_first_order)
 weak_form_residum= (
        u_trial * v_test * dx
                    +
                 time_step_length*dot(
                        grad(u_trial),
                        grad(v_test),
                    ) *dx

                    -(
                    u_old * v_test *dx
                    +time_step_length*heat_source*v_test*dx
                    ))
 weak_form_lhs=lhs(weak_form_residum)
 weak_form_rhs=rhs(weak_form_residum)
 u_solution =Function(lagrange_polynomials_space_first_order)
n_time_step=5
time_current =0.0
for i in range(n_time_step):
   time_current += time_step_length
   solve(
      weak_form_lhs==weak_form_rhs,
      u_solution,
      boundary_condition
   )
   u_old.assign(u_solution)
   plot(u_solution,label=f"t={time_current:1.1f}")

plt.legend()
plt.title("Heat conduction")
plt.xlabel("x position")
plt.ylabel("temperature")
plt.show()
