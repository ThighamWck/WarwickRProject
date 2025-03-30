"""
#################### 
# Backwards Committor Solver - for Fourth Year Project
# Author - Thomas Higham
# Date - 22/04/25
# University of Warwick
#####################
"""
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import autograd.numpy as anp
from autograd import elementwise_grad

def solve_backwards_committor_2D(phi, psi, bx, by, N):
    """
    Solve the 2D backward committor (steady advection-diffusion) equation with coordinate transform (x,y) -> (f(x,y), y):
        d_xx(u) + d_yy(u) - bx(x,y) * d_x(u) - by(x,y) * (d_y(u) + d_x(u)*f_y(x,y)) + d_yy(f)*d_x(u)
        + d_y(f)^2 * d_xx(u) + 2 * d_y(f) * d_yy(u) = 0
    on (0,1)^2 with Dirichlet BCs in x (u(0,y)=1, u(1,y)=0)
    and periodic BCs in y.

    Throughout I use the "ij" indexing convention.

    Parameters:
    -----------
    bx, by : functions bx(x,y), by(x,y) defining the advection velocity components
    N : int, Number of interior points in x-direction or y direction
    phi, psi  : left set, A, and right set, B, parametrisations in y.
                    used to define coordinate transform function f

    Returns:
    --------
    u : 2D numpy array containing the solution
    X, Y : Meshgrid for plotting
    """
    dx = 1.0 / (N + 1)
    dy = 1.0 / (N + 1)
 
    # Create grid
    x = anp.linspace(0, 1, N+2)  # Including boundary points
    y = anp.linspace(0, 1, N+2)  # Periodic in y
    X, Y = anp.meshgrid(x, y, indexing='ij')

    # Initialize solution matrix
    u = anp.zeros((N+2, N + 2))  # Including Dirichlet BCs
    
    def f_derivatives(phi, psi, X, Y):
        """Autograd-compatible derivatives"""
        def f(x, y):
            return (x - phi(y)) / (psi(y) - phi(y))
        
        # Vectorize the gradient calculations
        df_dy = elementwise_grad(f, argnum=1)
        df_dy2 = elementwise_grad(lambda x, y: df_dy(x, y), argnum=1)

        # Flatten the meshgrid arrays
        X_flat = X.flatten()
        Y_flat = Y.flatten()

        # Compute derivatives for all points at once on the flattened grid
        f_y_values_flat = df_dy(X_flat, Y_flat)  # First derivative
        f_y2_values_flat = df_dy2(X_flat, Y_flat)  # Second derivative

        # Reshape the results back into meshgrid shape
        f_y_values = f_y_values_flat.reshape(X.shape)
        f_y2_values = f_y2_values_flat.reshape(X.shape)
        
        return f_y_values, f_y2_values


    # Calculating f derivatives at grid points.
    f_y_values, f_y2_values = f_derivatives(phi, psi, X, Y)

    # Flatten the meshgrid arrays
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    #b is implemented in original coords so need to evaluate at Xdash, original coords associated with grid.
    Xdash_flat = phi(Y_flat) + X_flat*(psi(Y_flat) - phi(Y_flat))
    Xdash = Xdash_flat.reshape(X.shape)
    bx_values = bx(Xdash,Y)
    by_values = by(Xdash,Y)
    #Making sure vector field is same at both periodic boundaries
    bx_values[:,N+1] = bx_values[:, 0] 
    by_values[:, N+1]  = by_values[:,0]

    
    # Set up matrix system AU = B
    rows, cols, data = [], [], []
    B = anp.zeros(N*(N+2))

    """ We will use a sparse matrix form. The csr_matrix constructor takes the (rows, cols, data) lists and automatically sums duplicate entries for the same (row, column) pair.
        For loops set up system in matrix vector form. idx represents going along y-axis before incrementing x
        we range i from 0 to N+1 as we dont need to access the 0 and N+2 points as value is set. 
    """
    for i in range(1, N+1):
        #filling in array as though filling in matrix along rows
        for j in range(N+2):
            idx = (i-1)*(N+2) + j

            #Coordinate transform terms (backwards difference)
            f_y, f_yy = f_y_values[i,j], f_y2_values[i, j]

            # Diffusion terms - first we add the u_i_j point.(f_y)^2 term only appears in x direction.
            rows.append(idx); cols.append(idx); data.append(-2*(1 + (f_y)**2)/dx**2 - 2/dy**2)
            
            #Adding sides of stencil in x-direction so long as not near x boundary.
            if i > 1:
                rows.append(idx); cols.append(idx-(N+2)); data.append((1+f_y**2)/dx**2)
            if i < N:
                rows.append(idx); cols.append(idx+(N+2)); data.append((1+f_y**2)/dx**2)
            if i ==1:
                #Including effect of left boundary condition from diffusion in B
                B[idx] = B[idx] -(1+f_y**2)/dx**2
                
            
            # adding tops of stencil in y-direction. % allows for wrapping around to last y-point when j=0 and first when j=N+1
            # hence including periodic boundary conditions
            rows.append(idx); cols.append((i-1)*(N+2) + (j-1)%(N+1)); data.append(1/dy**2)
            rows.append(idx); cols.append((i-1)*(N+2) + (j+1)%(N+1)); data.append(1/dy**2)
            
            # Advection terms (backward difference)
            bx_ij, by_ij = -bx_values[i, j], -by_values[i, j]
            #bx_ij, by_ij = bx(x[i], y[j]), by(x[i], y[j])
            #Backwards difference x-axis -ve part. idx-(N+2) is adjacent point

                        # x-direction upwinding
            combined_x_advection = bx_ij + by_ij*f_y + f_yy
            if i > 1 and i < (N): #away from dirichlet boundaries
                if combined_x_advection >= 0:  # Flow is positive or zero - use backward difference
                    rows.append(idx); cols.append(idx-(N+2)); data.append(-combined_x_advection/dx)
                    rows.append(idx); cols.append(idx); data.append(combined_x_advection/dx)
                else:  # Flow is negative - use forward difference
                    rows.append(idx); cols.append(idx); data.append(-combined_x_advection/dx)
                    rows.append(idx); cols.append(idx+(N+2)); data.append(combined_x_advection/dx)
            elif i==1: #might have to interact with dirichlet boundary u(0,y)=0
                if combined_x_advection >=0: # Flow is positive or zero - use backward difference
                    # We force u_(0,j) =1 so include 1 into solution B. Already a zero there so we are good.
                    B[idx] = B[idx] + combined_x_advection/dx
                    rows.append(idx); cols.append(idx); data.append(combined_x_advection/dx)
                else:  # Flow is negative - use forward difference
                    rows.append(idx); cols.append(idx); data.append(-combined_x_advection/dx)
                    rows.append(idx); cols.append(idx+(N+2)); data.append(combined_x_advection/dx)
            elif i == (N):
                if combined_x_advection >=0: # Flow is positive or zero - use backward difference
                    rows.append(idx); cols.append(idx - (N+2)); data.append(-combined_x_advection/dx) 
                    rows.append(idx); cols.append(idx); data.append(combined_x_advection/dx)
                else:  # Flow is negative - use forward difference
                    rows.append(idx); cols.append(idx); data.append(-combined_x_advection/dx)
                    # We force u_(N+1,j) =0 - already a zero in B
                   
                
            # y-direction upwinding
            if by_ij >= 0:  # Flow is positive or zero - use backward difference
                rows.append(idx); cols.append((i-1)*(N+2) + (j-1)%(N+1)); data.append(-by_ij/dy)
                rows.append(idx); cols.append(idx); data.append(by_ij/dy)
            else:  # Flow is negative - use forward difference
                rows.append(idx); cols.append(idx); data.append(-by_ij/dy)
                rows.append(idx); cols.append((i-1)*(N+2) + (j+1)%(N+1)); data.append(by_ij/dy)

            #Mixed second difference d2u/dxdy.
            if i>1:
                rows.append(idx); cols.append((i-2)*(N+2) + (j-1)%(N+1)); data.append((2*f_y)/(4*dx*dy))
                rows.append(idx); cols.append((i-2)*(N+2) + (j+1)%(N+1)); data.append(-(2*f_y)/(4*dx*dy))
            if i < N:
                rows.append(idx); cols.append((i)*(N+2) + (j-1)%(N+1)); data.append(-(2*f_y)/(4*dx*dy))
                rows.append(idx); cols.append((i)*(N+2) + (j+1)%(N+1)); data.append((2*f_y)/(4*dx*dy))
            #At right boundary, conditions lead to mixed derivatives cancelling for i ==0. 0 otherwise for i==N+1

            

    # Create sparse matrix
    A = csr_matrix((data, (rows, cols)), shape=(N*(N+2), N*(N+2)))

    # Solve the system
    U = spsolve(A, B)

    # Reshape the solution
    u = U.reshape((N, N+2))
    u = anp.vstack((anp.ones(N+2), u, anp.zeros(N+2)))

    return u, X, Y
