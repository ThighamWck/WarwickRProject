"""
#################### 
# Turbulent velocity field - for Fourth Year Project
# Author - Thomas Higham and Tobias Grafke. Adapted from incompressible Navier Stokes code by Tobias.
# Date - 25/04/25
# University of Warwick
#####################
# """

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.interpolate import RegularGridInterpolator

def validate_resolution(N_x, N_y, Reynolds):
    """
    Validate if grid resolution is sufficient for given Reynolds number
    """
    N_min = int(np.ceil(Reynolds**(3/4)))
    if N_x < N_min or N_y < N_min:
        print(f"Warning: Grid resolution {N_x}x{N_y} may be insufficient for Re={Reynolds}")
        print(f"Recommended minimum resolution: {N_min}x{N_min}")
        return False
    return True

def turbulent_velocity_field(eps_1, eps_2, Reynolds, L, nu):
    """
    Generate a turbulent velocity field scaled for a specific Reynolds number
    
    Parameters:
    eps_1, eps_2 - domain parameters that define rectangular epsilon neighbourhood of Omega
    Reynolds - Reynolds number of the flow
    L - characteristic length scale
    nu - kinematic viscosity
    """
    # Validate resolution
    N_x, N_y = 311, 311
    # if not validate_resolution(N_x, N_y, Reynolds):
    #     N_x = N_y = int(np.ceil(Reynolds**(3/4)))
    #     print(f"Adjusting resolution to {N_x}x{N_y}")
    
    # ===========================
    # SET UP FOURIER SPACE
    # ===========================
    # Store characterisitc length L for Reynolds number calculation
    L_char = L
    
    # Domain size for spectral method
    L_domain = 1 + 2 * max(eps_1, eps_2)
    
    # Define grid and spacing in real space
    range_x, range_y = np.arange(N_x), np.arange(N_y)
    dx, dy = L_domain / N_x, L_domain / N_y
    xv, yv = dx * (range_x - 0.5 * N_x), dy * (range_y - 0.5 * N_y)  # Centered grid around (0,0)

    # Define grid and spacing in Fourier space
    k_xv = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)
    k_yv = 2 * np.pi * np.fft.fftfreq(N_y, d=dy)
    x, y = np.meshgrid(xv, yv, indexing='xy')
    kx, ky = np.meshgrid(k_xv, k_yv, indexing='ij')
    k2 = kx**2 + ky**2  # Squared wavenumber magnitude

    # ===========================
    # PROJECT TO DIVERGENCE-FREE SPACE
    # ===========================
    # Projection operator to ensure the velocity field is divergence-free
    def P(vx, vy):
        # Avoid division by zero at k=0
        k2_safe = k2.copy()
        k2_safe[0, 0] = 1.0
        
        # Compute divergence in Fourier space
        v = -(1j * kx * vx + 1j * ky * vy) / k2_safe
        v[0, 0] = 0  # Set zero mode to 0
        
        # Project to divergence-free space
        return vx - 1j * kx * v, vy - 1j * ky * v

    # ===========================
    # INITIALIZE VELOCITY FIELD
    # ===========================
    # Define wavenumber and Nyquist limit
    k = np.sqrt(k2)
    k_nyquist = np.pi * N_x / L_domain
    
    # Select modes based on physical scales
    kfmin = 2 * np.pi / L_domain  # Largest scale ~ domain size
    kfmax = min(k_nyquist / 2, 20 * kfmin)  # Limit by resolution and reasonable range
    
    # Use logarithmic spacing for mode selection
    log_k = np.log10(k + 1e-10)  # Avoid log(0) errors
    log_kfmin = np.log10(kfmin)
    log_kfmax = np.log10(kfmax)
    
    # Select modes in logarithmic bands
    initModes = (log_k >= log_kfmin) & (log_k <= log_kfmax)
    
    n_active = np.sum(initModes)
    # print(f"Active modes: {n_active}")
    # print(f"Wavenumber range: {kfmin:.2f} to {kfmax:.2f}")
    # print(f"Number of modes in each dimension: {int(np.sqrt(n_active))}")
    
    # if n_active < 10:
    #     raise ValueError(f"Too few modes ({n_active}). Try increasing resolution.")
    
    # Create random Fourier modes with larger initial amplitude
    ux = np.zeros((N_x, N_y), dtype=complex)
    uy = np.zeros((N_x, N_y), dtype=complex)
    
    # Increase initial amplitude to ensure non-zero values
    amplitude = 1e4
    ux[initModes] = amplitude * (np.random.randn(np.sum(initModes)) + 1j * np.random.randn(np.sum(initModes)))
    uy[initModes] = amplitude * (np.random.randn(np.sum(initModes)) + 1j * np.random.randn(np.sum(initModes)))

    # Modified energy spectrum scaling
    scaling = np.zeros_like(k)
    mask = (k > 0) & initModes
    scaling[mask] = k[mask]**(-5/3)  # Use k^(-5/3) scaling for 2D turbulence
    
    # Add small constant to prevent division by zero
    scaling = scaling + 1e-10
    
    # Apply scaling to Fourier coefficients
    ux *= scaling
    uy *= scaling

    # Ensure field is divergence-free
    ux, uy = P(ux, uy)

    # Transform back to physical space
    ux_physical = np.real(ifft2(ux))
    uy_physical = np.real(ifft2(uy))

    # ===========================
    # NORMALIZE TO TARGET RMS
    # ===========================
    #rms_target = np.sqrt(Reynolds * nu / L_char)  # Corrected RMS velocity target
    rms_target = (Reynolds * nu) / L_char

    current_rms = np.sqrt(np.mean(ux_physical**2 + uy_physical**2))
    
    # if current_rms < 1e-10:
    #     raise ValueError(f"RMS velocity too small ({current_rms}). Try increasing initial amplitude.")
    
    # Scale velocity field to match required Reynolds number
    ux_physical = ux_physical * (rms_target / current_rms)
    uy_physical = uy_physical * (rms_target / current_rms)
    
    # Verify final Reynolds number
    final_rms = np.sqrt(np.mean(ux_physical**2 + uy_physical**2))
    actual_reynolds = (final_rms * L_char) / nu
    # print(f"Target Reynolds number: {Reynolds}")
    # print(f"Actual Reynolds number: {actual_reynolds:.4f}")

    # # ===========================
    # # VISUALIZE RESULTS
    # # ===========================
    # plt.figure(figsize=(12, 10))

    # # Velocity magnitude
    # velocity_mag = np.sqrt(ux_physical**2 + uy_physical**2)
    # plt.subplot(2, 2, 1)
    # plt.pcolormesh(x, y, velocity_mag, cmap='viridis', shading='auto')
    # plt.colorbar(label='Velocity Magnitude')
    # plt.title('Velocity Magnitude')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('equal')
    # plt.savefig("turbulence.png", dpi=300, bbox_inches="tight")


    # ===========================
    # INTERPOLATOR SETUP
    # ===========================
    x = np.linspace(-max(eps_1, eps_2) - 0.01, 1 + max(eps_1, eps_2) + 0.01, N_x)
    y = np.linspace(0, 1, N_y)

    # Create interpolators for each component of the vector field
    interp_ux = RegularGridInterpolator((x, y), ux_physical, method='cubic')
    interp_uy = RegularGridInterpolator((x, y), uy_physical, method='cubic')
    




    return interp_ux, interp_uy
