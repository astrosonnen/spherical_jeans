import numpy as np
from spherical_jeans import sigma_model, tracer_profiles
from spherical_jeans.mass_profiles import powerlaw


# this code calculates the line-of-sight stellar velocity dispersion of a galaxy with a power-law density profile, tracers decsribed by a Jaffe profile and isotropic orbits, within a circular aperture of radius Re/2.

gamma = 2. # radial slope of the power-law density profile: $rho(r) \propto r^{-\gamma}$
Reff = 5. # half-light radius in kpc
aperture = 0.5*Reff # radius of circular aperture within which the velocity dispersion is calculated
Meff = 3.*1e11 # total projected mass enclosed within the half-light radius (used to normalize the density profile)

a_Jaf = Reff * 1.3428 # value of 'a' parameter of a Jaffe profile corresponding to a half-light radius Reff
a_Her = Reff * 0.5509 # same for a Hernquist profile

# physical constants in cgs units
kpc = 3.08568025e21
G = 6.67300e-8
M_Sun = 1.98892e33

def Mass_profile(r):
    norm = Meff/powerlaw.M2d(Reff, gamma)
    return norm * powerlaw.M3d(r, gamma)

s2 = sigma_model.sigma2(Mass_profile, aperture, a_Jaf, tracer_profiles.jaffe)

sigma = (s2*G*M_Sun/kpc)**0.5/1e5 # velocity dispersion in km/s

print sigma

# now let's keep the normalization fixed and vary the density slope

gammas = [1.8, 2., 2.2]

for gamma in gammas:
    def Mass_profile(r):
        norm = Meff/powerlaw.M2d(Reff, gamma)
        return norm * powerlaw.M3d(r, gamma)
    
    s2 = sigma_model.sigma2(Mass_profile, aperture, a_Jaf, tracer_profiles.jaffe)
    sigma = (s2*G*M_Sun/kpc)**0.5/1e5 # velocity dispersion in km/s
    print sigma

