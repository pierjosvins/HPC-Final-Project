# Solving 2D Navier Stokes equation
 ## Students
  * Youssef HOURRI    
  * Pierjos Francis COLERE MBOUKOU
 
 ## Supervisors
  * Imad KISSAMI
  * Nouredine OUHADDOU
 
 ## TODO
   1. Solving Poisson equation using Finite difference and Jacobi’s iterative solver, detail on the slide 2.
   2. Solving the advection diffusion equation using Finite difference.
   3. Coupling both to solve the NS equation (already done using Numba and Pyccel).
   4. Compare the numerical result with the serial code, and show the execution time for the parallel code using 1, 2, 4 ... processes.
   5. Compare the results with the OpenMP implementation.
   
      D_u = f(x,y)= 2*(x*x-x+y*y -y)
      u equal 0 on the boudaries
      The exact solution is u = x*y*(x-1)*(y-1)
 
    The u value is :
      coef(1) = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
      coef(2) = 1./(hx*hx)
      coef(3) = 1./(hy*hy)
      
      u(i,j)(n+1)=coef(1)∗(coef(2)∗(u(i+1,j)+u(i-1,j))+ coef(3)∗(u(i,j+1)+u(i,j−1))−f(i,j))

