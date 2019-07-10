import numpy as np
from spherical_jeans import sigma_model, tracer_profiles
from spherical_jeans.mass_profiles import powerlaw


# this code calculates the line-of-sight stellar velocity dispersion of a galaxy with a power-law density profile, tracers decsribed by a Jaffe profile and orbits with a constant anisotropy profile, within a circular aperture of radius Re/2.

gamma = 2. # radial slope of the power-law density profile: $rho(r) \propto r^{-\gamma}$
Reff = 5. # half-light radius in kpc
aperture = 0.5*Reff # radius of circular aperture within which the velocity dispersion is calculated
Meff = 3.*1e11 # total projected mass enclosed within the half-light radius (used to normalize the density profile)

a_Jaf = Reff * 1.3428 # value of 'a' parameter of a Jaffe profile corresponding to a half-light radius Reff

# 
# physical constants in cgs units
kpc = 3.08568025e21
G = 6.67300e-8
M_Sun = 1.98892e33

def Mass_profile(r):
    norm = Meff/powerlaw.M2d(Reff, gamma)
    return norm * powerlaw.M3d(r, gamma)

betas = [-0.2, 0., 0.2] # list of anisotropy parameter values
for beta in betas:
    s2 = sigma_model.sigma2(Mass_profile, aperture, a_Jaf, tracer_profiles.jaffe, anisotropy='beta', anis_par=beta)
    sigma = (s2*G*M_Sun/kpc)**0.5/1e5 # velocity dispersion in km/s
    print sigma

