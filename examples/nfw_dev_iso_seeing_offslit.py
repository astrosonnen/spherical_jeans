import numpy as np
from spherical_jeans import sigma_model, tracer_profiles
from spherical_jeans.mass_profiles import nfw


# this code calculates the seeing-convolved, surface brightness weightedd line-of-sight stellar velocity dispersion of a galaxy with a Navarro, Frenk & White (NFW) density profile, tracers decsribed by a spherically deprojected de Vaucouleurs profile and isotropic orbits, within a rectangular aperture of width 0.5Reff and height 1.0Reff, offset from the galaxy center.

Reff = 5. # half-light radius in kpc
aperture = [Reff, 1.5*Reff, -0.5*Reff, 0.5*Reff] # coordinates (x1, x2, y1, y2) of the vertices of the rectangular aperture within which the velocity dispersion is calculated
seeing = 0.3*Reff # seeing FWHM

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
    
    s2 = sigma_model.sigma2(Mass_profile, aperture, Reff, tracer_profiles.deVaucouleurs, seeing=seeing)
    
    sigma = (s2*G*M_Sun/kpc)**0.5/1e5 # velocity dispersion in km/s
    
    print sigma

