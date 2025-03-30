import numpy as np
import pandas as pd
import autograd.numpy as anp
from Finite_difference_forward_comittor import solve_forward_committor_2D
from Finite_difference_backwards_committor import solve_backwards_committor_2D
from Trig_polynomial_boundary import generate_trig_functions
from turbulent_velocity_field import turbulent_velocity_field

def generate_training_2(num_solutions, grid_size):
    """
    This code generates training data for my committor function problem on a special domain that is transformed to the square [0,1]^2.
    Training data to be used for a FNO.
    """

    #Setting up data list
    data_list = []
    #random seed to make training data reproducible
    np.random.seed(220325)  #training data
    #np.random.seed(2203252)  #test data
    #Setting minimal distance between boundary parametrisations phi and psi
    eps_trig = 0.5
    for solution_id in range(num_solutions):
        #progress tracking
        print(f"Generating solution {solution_id+1}/{num_solutions}")
        #Create grid
  
        #X, Y = np.meshgrid(np.linspace(0, 1, grid_size*20), np.linspace(0, 1, grid_size*20), indexing='ij')
        #generate domain shape via phi and psi
        while True:
        # Generate boundaries via parametrisations phi and psi
            phi, psi = generate_trig_functions(Print=False)
            
            # Check separation across the domain
            y_vals = np.linspace(0, 1, 400)
            x_vals1 = anp.vectorize(phi)(y_vals)
            x_vals2 = anp.vectorize(psi)(y_vals)
            
            # Check if adequate separation exists everywhere
            if np.all(x_vals2 > x_vals1 + eps_trig):
                #Finding episilon neighbourhoods for generating turbulence in domain (generation is on a super-square).
                # Get values less than or equal to 0
                negative_vals_1 = x_vals1[x_vals1 <= 0]
                vals_2 = x_vals2[x_vals2>=1]
                # Find the largest in magnitude (which is closest to 0)
                largest_neg = negative_vals_1[np.argmax(np.abs(negative_vals_1))]
                largest_pos = vals_2[np.argmax(np.abs(vals_2))]
                #Find the largest in magnitude 
                eps_1 = -largest_neg
                eps_2 = largest_pos-1
                break  # Found non-overlapping functions

        #define f and finv
        def f(x, y):
            return (x - phi(y)) / (psi(y) - phi(y))
        def finv(x,y):
            X_flat = x.flatten()
            Y_flat = y.flatten()
            #b is implemented in original coords so need to evaluate at Xdash, original coords associated with grid.
            Xdash_flat = phi(Y_flat) + X_flat*(psi(Y_flat) - phi(Y_flat))
            Xdash = Xdash_flat.reshape(x.shape)
            return Xdash
                
        #generate random vector field b(x,y) and define btilde(f(x,y),y) = b(f^{-1}(x,y),y) so in correct coords
        b_x , b_y = turbulent_velocity_field(eps_1, eps_2, Reynolds = 10, L = 0.2, nu = 1)
        def btilde_x(x,y):
            # Reshape meshgrid points into format needed by RegularGridInterpretor
            points = anp.vstack(( x.flatten(), y.flatten())).T
            return b_x(points).reshape(x.shape)
        def btilde_y(x,y):
             # Reshape meshgrid points into format needed by RegularGridInterpretor
            points = anp.vstack(( x.flatten(), y.flatten())).T
            return b_y(points).reshape(x.shape)
        
        # reason for this is step size is 1/15 between [0,1] as (0, 1/15,... 14/15, 1).
        # Define the 311x311 grid - easy to identify with a coarser 32 x 32 grid when training.
        n_fine = (grid_size-1)*5 + 1

        #Solving for forward committor
        qplus, X1, Y1 = solve_forward_committor_2D(phi, psi, btilde_x, btilde_y, N= (n_fine-2)) #Nx and Ny count num of interior points
        #Solving for backward committor
        qminus, X2, Y2 = solve_backwards_committor_2D(phi, psi, btilde_x, btilde_y, N = (n_fine-2))
        #Committor function
        rho = qplus * qminus
      # Create a meshgrid of coarse indices 
        coarse_indices = np.linspace(0, n_fine - 1, grid_size, dtype=int)
        coarse_indices_i, coarse_indices_j = np.meshgrid(coarse_indices, coarse_indices)

        # Extract the coarse grid from X and Y
        coarse_grid_x = X1[coarse_indices_i, coarse_indices_j]
        coarse_grid_y = Y1[coarse_indices_i, coarse_indices_j]

        #Storing data in dataframe. We only want every 20 points.
        for i in range(grid_size):
            for j in range(grid_size):
                x_ij = coarse_grid_x[i,j]
                y_ij = coarse_grid_y[i,j]
                finv_val = finv(x_ij, y_ij)
                # need to evaluate at finv(x,y). This is done normally inside solve functions.
                b1, b2 = btilde_x(finv_val, y_ij) , btilde_y(finv_val, y_ij)
               

                # Get the u value at the corresponding indices in the original array
                rho_val = rho[coarse_indices_i[i,j], coarse_indices_j[i,j]]
                data_list.append([solution_id, x_ij, y_ij, b1, b2, finv_val, rho_val])

    # Create DataFrame
    df = pd.DataFrame(data_list, columns=["solution_id", "x", "y", "b1", "b2", "finv", "rho"])

    # Save to CSV
    df.to_csv("train_data_63_transform_5000.csv", index=False)


    return