# Spherical Jeans modeling

This code calculates the surface brightness-weighted, seeing-convolved (optional) line-of-sight stellar velocity dispersion integrated within a rectangular or circular aperture of a galaxy, integrating the spherical Jeans equation.

## Model assumptions

- Spherically symmetric stellar system
- Time-independent
- The stellar distribution can be described by the collisionless Boltzmann equation

## Model ingredients

- 3D distribution of tracers (i.e. stellar particles) and its projection on the plane of the sky: $\rho_*(r)$, $I(R)$.
- 3D mass distribution of the galaxy: $M(r)$.
- 3D orbital anisotropy parameter: $\beta$.
- Aperture within which to average the line-of-sight velocity dispersion
- Seeing FWHM (optional)

## The code in a nutshell

The function `sigma2` in `sigma_model.py` calculates the velocity dispersion. It does so by, first, calculating the surface brightness-weighted line-of-sight velocity dispersion as a function of position in the sky (we refer to Appendix 2 of [Mamon & Lokas 2005](https://ui.adsabs.harvard.edu/abs/2005MNRAS.363..705M/) for a derivation), then convolving this by a Gaussian point spread function (if provided), and finally by integrating the result over the aperture provided.

The code assumes that all sizes are in the same physical units (including the seeing FWHM). It returns $\sigma^2/G$ in the same units as $M(r)/r$, where $M(r)$ is the galaxy mass enclosed within a shell of radius $r$, which must be specified when calling `sigma2`.

## Examples

The folder `examples/` contains a few scripts that calculate the velocity dispersion for a variety of cases.

1. `powerlaw_jaffe_iso.py`: Power-law density profile, tracers following a Jaffe profile, isotropic orbits, no seeing, circular aperture of radius Re/2.
2. `powerlaw_jaffe_beta.py`: Power-law density profile, tracers following a Jaffe profile , constant positive (i.e. radial) anisotropy, no seeing, circular aperture of radius Re/2.
3. `nfw_dev_iso.py`: Navarro, Frenk & White (NFW) mass profile, spherically deprojected de Vaucouleurs tracer profile, isotropic orbits, no seeing, circular aperture of radius Re/2.
4. `nfw_dev_iso_seeing.py`: Navarro, Frenk & White (NFW) mass profile, spherically deprojected de Vaucouleurs tracer profile, isotropic orbits, non-zero seeing, circular aperture of radius Re/2.
5. `nfw_dev_iso_seeing_slit.py`: Navarro, Frenk & White (NFW) mass profile, spherically deprojected de Vaucouleurs tracer profile, isotropic orbits, non-zero seeing, rectangular aperture centered on the galaxy center.
6. `nfw_dev_iso_seeing_offslit.py`: Navarro, Frenk & White (NFW) mass profile, spherically deprojected de Vaucouleurs tracer profile, isotropic orbits, non-zero seeing, rectangular aperture offset from the galaxy center.
7. `sersic_sersic_iso.py`: Spherically deprojected Sersic profile for both the mass and the tracer distribution (i.e. mass-follows-light), isotropic orbits, no seeing, circular aperture of radius Re/2.

