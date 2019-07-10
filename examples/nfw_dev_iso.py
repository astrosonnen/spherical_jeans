import numpy as np
from spherical_jeans import sigma_model, tracer_profiles
from spherical_jeans.mass_profiles import nfw


# this code calculates the line-of-sight stellar velocity dispersion of a galaxy with a Navarro, Frenk & White (NFW) density profile, tracers decsribed by a spherically deprojected de Vaucouleurs profile and isotropic orbits, within a circular aperture of radius Re/2.

Reff = 5. # half-light radius in kpc
aperture = 0.5*Reff # radius of circular aperture within which the velocity dispersion is calculated
Meff = 3.*1e11 # total projected mass enclosed within the half-light radius (used to normalize the density profile)

# physical constants in cgs units
kpc = 3.08568025e21
G = 6.67300e-8
M_Sun = 1.98892e33

rss = [10., 30., 100.] # scale radius of the NFW profile, in kpc

for rs in rss:
    def Mass_profile(r):
        norm = Meff/nfw.M2d(Reff, rs)
        return norm * nfw.M3d(r, rs)
    
    s2 = sigma_model.sigma2(Mass_profile, aperture, Reff, tracer_profiles.deVaucouleurs)
    
    sigma = (s2*G*M_Sun/kpc)**0.5/1e5 # velocity dispersion in km/s
    
    print sigma

