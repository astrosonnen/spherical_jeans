import numpy as np
from scipy.integrate import quad
import os
from scipy.special import gamma as gfunc
from scipy.interpolate import splrep, splint, splev


def b(nser):
    return 2*nser - 1./3. + 4/405./nser + 46/25515/nser**2

def L(reff, nser):
    return reff**2*2*np.pi*nser/b(nser)**(2*nser)*gfunc(2*nser)

def Sigma(R, reff, nser): # projected surface mass density
    return np.exp(-b(nser)*(R/reff)**(1./nser))/L(nser, reff)

def rho(r, reff, nser): # spherical deprojection

    deriv = lambda R: -b(nser)/nser*(R/(reff))**(1/nser)/R*Sigma(R, reff, nser)
    return -1./np.pi*quad(lambda R: deriv(R)/(R**2 - r**2)**0.5, r, np.inf)[0]

def get_m3d_spline(reff, nser, rmin=1e-4, rmax=1e4, nr=1001):

    # deprojects light profile and integrates it to obtain 3d stellar mass profile
    r_grid = np.logspace(np.log10(rmin), np.log10(rmax), nr)
    rho_grid = np.zeros(nr)
    for i in range(nr):
        rho_grid[i] = rho(r_grid[i], reff, nser)
    rs0 = np.append(0., r_grid)
    mp0 = np.append(0., 4.*np.pi*rho_grid*r_grid**2)
    
    mprime_spline = splrep(rs0, mp0)
    
    m3d_grid = np.zeros(nr+1)
    for i in range(nr):
        m3d_grid[i+1] = splint(0., r_grid[i], mprime_spline)
    
    m3d_spline = splrep(np.append(0., r_grid), m3d_grid)
    return m3d_spline

def M3d(r, reff, nser, m3d_spline=None):
    if m3d_spline is None:
        m3d_spline = get_m3d_spline(reff, nser)

    return splev(r, m3d_spline)

