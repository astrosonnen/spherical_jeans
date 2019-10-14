import numpy as np
from spherical_jeans import sigma_model, tracer_profiles
from spherical_jeans.mass_profiles import sersic
from scipy.interpolate import splev


# calculates the line-of-sight stellar velocity dispersion of a galaxy with a spherically deprojected de Vaucouleurs mass and tracer profile (mass-follows-light) and isotropic orbits, within a circular aperture of radius Re/2.

Reff = 5. # half-light radius in kpc
aperture = 0.5*Reff # radius of circular aperture within which the velocity dispersion is calculated

Meff = 1e11 # total projected mass enclosed within the half-light radius (used to normalize the density profile)

# physical constants in cgs units
kpc = 3.08568025e21
G = 6.67300e-8
M_Sun = 1.98892e33

nser_list = [1., 3., 5.] # list of values of the Sersic index

for nser in nser_list:

    # deprojects a Sersic profile and obtains a spline interpolation of the M(r) profile
    M3d_spline = sersic.get_m3d_spline(nser, Reff)
    norm = Meff/sersic.M2d(Reff, nser, Reff)

    def Mass_profile(r):
        return norm*splev(r, M3d_spline)

    s2 = sigma_model.sigma2(Mass_profile, aperture, [Reff, nser], tracer_profiles.sersic)

    sigma = (s2*G*M_Sun/kpc)**0.5/1e5 # velocity dispersion in km/s
    
    print sigma

