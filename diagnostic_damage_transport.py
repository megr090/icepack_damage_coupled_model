# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

"""Description of the diagnostic damage model

This module contains a solver for the diagnostic damage model
based on the model from (Ranganathan, Robel, Huth, Duddu in prep.)

"""

from operator import itemgetter
import icepack
import numpy as np
import ufl
import firedrake
from firedrake import interpolate, conditional, sqrt, max_value, Constant
from icepack.constants import year, ideal_gas as R, glen_flow_law as n, strain_rate_min, ice_density, water_density, gravity
from icepack.calculus import sym_grad, trace, Identity
from icepack.models.hybrid import horizontal_strain_rate, vertical_strain_rate, stresses
from icepack.mask_smooth_functions import firedrakeSmooth

damage_model = icepack.models.DamageTransportNoSrc()
damage_solver = icepack.solvers.DamageSolver(damage_model)

pressure_type = "ice"

def calcAdvectionDiagnosticDamage(**kwargs):
 
    """
    This function calculates diagnostic damage with an advective term
    
    Inputs: 
    ----------
    u: 2D velocity (m/yr)
    s: 2D surface elevation (m)
    h: 2D thickness (m)
    Aglen: fluidity in 2D
    num_layers: number of z layers
    criterion: the stress criterion for damage (options = vms for von Mises 
        stress, mps for maximum principal stress, hayhurst for the Hayhurst stress)
    K,σ_0: parameters in the stress threshold equation
    Dold: damage field from the previous timestep
    Dmax: maximum value of the damage field
    grainsize: grain size (mm)
    
    Outputs: 
    ----------
    D3: 3D damage
    """

    keys = ("dt", "velocity", "thickness", "surface", "fluidity", "num_layers", "criterion", "threshold", "Dold", "Dmax")
    dt, u, h, s, Aglen, num_layers, criterion, σ_t, Dold, Dmax = itemgetter(*keys)(kwargs)

    # define 2D mesh   
    mesh2d = h.function_space().mesh()
    Q2 = h.function_space()

    # calculate advection of 2D damage with no source
    Doldplus = damage_solver.solve(
        dt,
        damage=Dold,
        velocity=u,
        damage_inflow=Constant(0.0),
    )
    Doldplus = interpolate(conditional(Doldplus > Dmax,Dmax,Doldplus),Q2)

    # define 3D mesh
    mesh3d = firedrake.ExtrudedMesh(mesh2d, layers=num_layers)
    Q3 = firedrake.FunctionSpace(mesh3d, family='CG', degree=2, vfamily='DG', vdegree=0)
    V3 = firedrake.VectorFunctionSpace(mesh3d, dim=2, family='CG', degree=2, vfamily='GL', vdegree=0)
    x3, y3, ζ = firedrake.SpatialCoordinate(mesh3d)

    # define 3D fluidity
    Aglen3D = firedrake.Function(Q3)
    Aglen_data = Aglen.dat.data_ro[:]
    Aglen3D.dat.data[:] = np.repeat(Aglen_data, num_layers, axis=0)
    Aglen = interpolate(Aglen3D,Q3)

    # find stress threshold, extrude to 3D
    σ_t = interpolate(σ_t,Q2)
    threshold_data = σ_t.dat.data_ro[:]
    threshold_3d = firedrake.Function(Q3)
    threshold_3d.dat.data[:] = np.repeat(threshold_data, num_layers, axis=0)

    # calculate diagnostic damage
    u3, s3, h3 = elevateTo3D(velocity = u, surface = s, thickness = h, num_layers = num_layers, Q3 = Q3, V3 = V3)
    stress = calcStress(thickness = h3, surface = s3, velocity = u3, fluidity = Aglen3D, criterion = criterion, vertical_coordinate = ζ, pressure_type = pressure_type)
    Ddiag = damage_diagnostic(thickness = h, stress = stress, threshold = threshold_3d, Dmax = Dmax)

    # find depth-averaged damage
    Dnew = firedrake.interpolate(conditional(Ddiag < Doldplus,Doldplus,Ddiag),Q2)
    D = firedrake.interpolate(conditional(Dnew > Dmax,Dmax,Dnew),Q2)
    
    return D

def damage_diagnostic(**kwargs):

    """
    This function calculates diagnostic damage, which sets damage to maximum everywhere where stress exceeds a threshold (in 3D)
    
    Inputs: 
    ----------
    thickness: 2D glacier thickness
    stress: chosen form of stress for damage
    threshold: the value of the stress threshold
    Dmax: maximum value of 2D damage
    
    Outputs: 
    ----------
    D3: 3D damage
    """
    
    keys = ("thickness", "stress", "threshold", "Dmax")
    h, stress, threshold, Dmax = itemgetter(*keys)(kwargs)

    Q3 = stress.function_space()
    Q2 = h.function_space()
    
    # elevate to 3D
    stress = interpolate(stress, Q3)
    D3 = firedrake.Function(Q3)
    D3 = firedrake.interpolate(conditional(stress > threshold,1,0),Q3)

    D2 = icepack.depth_average(D3)
    D2 = conditional(D2 < 0.00001, 0, D2)
    D2 = conditional(D2 > Dmax, Dmax, D2)
    D2 = interpolate(D2, Q2)

    return D2

def calcStress(**kwargs):
   
    """
    This function calculates different 3D stresses based on the input
    
    Inputs: 
    ----------
    h: 3D thickness (m)
    s: 3D surface elevation (m)
    u: 3D velocity (m/yr)
    A: fluidity
    criterion: the stress criterion for damage (options = vms for von Mises 
    stress, mps for maximum principal stress, hayhurst for the Hayhurst stress)
    ζ: z-coordinate
    
    Outputs: 
    ----------
    chosen stress field    
    """
    
    keys = ("thickness", "surface", "velocity", "fluidity", "criterion", "vertical_coordinate", "pressure_type")
    h, s, u, Aglen, criterion, ζ, p_type = itemgetter(*keys)(kwargs)
    Q3 = Aglen.function_space()

    # define z coordinate, where the first element in the array is the ice-bed interface (or ice-water interface) 
    zp = h * ζ + (s-h)
    zsl = 0

    # Calculate deviatoric stress
    ε_x, ε_z = horizontal_strain_rate(velocity=u, thickness=h, surface=s), vertical_strain_rate(velocity=u,
                                                                                                   thickness=h)
    τ_x, τ_z = stresses(strain_rate_x=ε_x, strain_rate_z=ε_z, fluidity=Aglen)

    # build effective deviatoric stress tensor
    t11 = τ_x[0, 0]
    t22 = τ_x[1, 1]
    t33 = τ_z[0]
    t12 = τ_x[0, 1]

    # Calculate ice pressure
    sigma_zz = ice_density * gravity * (s - zp)
    p_i = sigma_zz - t11 - t22

    #if p_type == "effective":
    #    # Calculate water pressure
    #    zdiff = interpolate(conditional(zp<zsl,1,0),Q3) # find regions where height does not exceed the water line
    #    p_w = firedrakeSmooth(interpolate((water_density*gravity*(zsl-zp))*zdiff,Q3),α=2e3)
    #    p_w = interpolate((water_density*gravity*(zsl-zp))*zdiff,Q3)
    #    
    #    # Calculate effective pressure
    #    peff = interpolate(p_i - p_w,Q3)
    #
    #    p = peff
    #else:
    #    p = p_i 

    # calculate Cauchy stress
    s11, s22, s33, s12 = t11 - p_i, t22 - p_i, t33 - p_i, t12

    # calculate invariants
    I1 = s11 + s22 + s33  # effective I1 invariant
    J2 = 0.5 * (t11 ** 2 + t22 ** 2 + t33 ** 2) + t12 ** 2  # effective J2 invariant
    vms = sqrt(3 * J2)  # effective von Mises stress

    lam1 = s33
    lam2 = 0.5 * (s11 + s22 + sqrt(s11 ** 2 - 2 * s11 * s22 + 4 * (s12 ** 2) + s22 ** 2))
    mps = max_value(lam1, lam2)  # effective max principal stress

    # effective Hayhurst stress
    alpha = 0.21  # weight of max principal stress in Hayhurst criterion
    beta = 0.63  # weight of von Mises stress in Hayhurst criterion

    chi = alpha * mps + beta * vms + (1 - alpha - beta) * I1

    if criterion == 'vms':
        return interpolate(vms,Q3)
    elif criterion == 'mps':
        return interpolate(mps,Q3)
    elif criterion == 'hayhurst':
        return interpolate(chi,Q3)
    else:
        print("Incorrect stress criterion selected")

def elevateTo3D(**kwargs):

    """
    This function elevates velocity, surface elevation, and thickness fields into 3D with num_layers dictating the number of layers
    
    Inputs: 
    ----------
    u: 2D velocity (m/yr)
    s: 2D surface elevation (m)
    h: 2D thickness (m)
    num_layers: number of z layers
    V3, Q3: vector and scalar function spaces on a 3D mesh
    
    Outputs: 
    ----------
    u3: 3D velocity (m/yr)
    s3: 3D surface elevation (m/yr)
    h3: 3D thickness (m/yr)
    
    """
    
    keys = ("velocity", "surface", "thickness", "num_layers", "Q3", "V3")
    u, s, h, num_layers, Q3, V3 = itemgetter(*keys)(kwargs)

    
    # elevate to 3D
    U = u.dat.data_ro[:]
    S = s.dat.data_ro[:]
    H = h.dat.data_ro[:]

    u3 = firedrake.Function(V3)
    s3 = firedrake.Function(Q3)
    h3 = firedrake.Function(Q3)

    u3.dat.data[:] = np.repeat(U, num_layers, axis=0)
    s3.dat.data[:] = np.repeat(S, num_layers, axis=0)
    h3.dat.data[:] = np.repeat(H, num_layers, axis=0)

    return u3, s3, h3

def calcStressThreshold(**kwargs):
    """
    This function calculates the stress threshold based on a general relationship where stress threshold 
    is proportional to the inverse square root of grain size.
    
    Inputs: 
    ----------
    K: constant (or fracture toughness) (MPa m^(1/2))
    σ_0: y-intercept (MPa)
    grainsize: grain size (m)
    
    Outputs: 
    ----------
    σ_t: fracture threshold    
    """
    
    keys = ("K", "σ_0", "grainsize")
    K, σ_0, grainsize = itemgetter(*keys)(kwargs)

    Q = grainsize.function_space()
    σ_t = interpolate(K*(grainsize)**(-1/2) + σ_0,Q)
    
    return σ_t