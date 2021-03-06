import numpy as np
from scipy.integrate import quad
import os
from scipy.special import gamma as gfunc
from scipy.interpolate import splrep, splint, splev


def b(nser):
    return 2*nser - 1./3. + 4/405./nser + 46/25515/nser**2

def L(nser, reff):
    return reff**2*2*np.pi*nser/b(nser)**(2*nser)*gfunc(2*nser)

def Sigma(R, nser, reff): # projected surface mass density
    return np.exp(-b(nser)*(R/reff)**(1./nser))/L(nser, reff)

def rho(r, nser, reff): # spherical deprojection

    deriv = lambda R: -b(nser)/nser*(R/(reff))**(1/nser)/R*Sigma(R, nser, reff)
    return -1./np.pi*quad(lambda R: deriv(R)/(R**2 - r**2)**0.5, r, np.inf)[0]

def get_m3d_spline(nser, reff, rmin=1e-4, rmax=1e4, nr=1001):

    # deprojects light profile and integrates it to obtain 3d stellar mass profile
    r_grid = np.logspace(np.log10(rmin), np.log10(rmax), nr)
    rho_grid = np.zeros(nr)
    for i in range(nr):
        rho_grid[i] = rho(r_grid[i], nser, reff)
    rs0 = np.append(0., r_grid)
    mp0 = np.append(0., 4.*np.pi*rho_grid*r_grid**2)
    
    mprime_spline = splrep(rs0, mp0)
    
    m3d_grid = np.zeros(nr+1)
    for i in range(nr):
        m3d_grid[i+1] = splint(0., r_grid[i], mprime_spline)
    
    m3d_spline = splrep(np.append(0., r_grid), m3d_grid)
    return m3d_spline

def M3d(r, nser, reff, m3d_spline=None):
    if m3d_spline is None:
        m3d_spline = get_m3d_spline(nser, reff)

    return splev(r, m3d_spline)

def M2d(R, nser, reff):
    R = np.atleast_1d(R)
    out = 0.*R
    for i in range(len(R)):
        out[i] = 2*np.pi*quad(lambda r: r*Sigma(r, nser, reff), 0., R[i])[0]
    return out

