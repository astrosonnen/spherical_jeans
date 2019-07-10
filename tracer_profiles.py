from math import pi
import numpy as np
import spherical_jeans
from scipy.interpolate import splev, splrep
from scipy.integrate import quad
from scipy.special import gamma as gfunc
import pickle


sjpath = spherical_jeans.__path__[0]

f = open('%s/deV_rho.dat'%sjpath,'r')
deV_rho_spline = pickle.load(f)
f.close()

def hernquist_sb(rvals,a):
    """
    Calculate the projected Hernquist surface brightness distribution.

    rvals - locations to evaluate the model
    a     - scale radius
    """

    from scipy import log,arccos
    s0 = np.atleast_1d(abs(rvals)/a)
    sbmodel = s0*0.
    s = s0[s0<1.]
    sbmodel[s0<1.] = log((1.+(1.-s**2)**0.5)/s)/(1.-s**2)**0.5
    s = s0[s0>1.]
    sbmodel[s0>1.] = arccos(1./s)/(s**2 - 1.)**0.5
    s = s0.copy()
    sbmodel *= (2.+s**2)
    sbmodel -= 3.
    sbmodel /= (a**2)*(1.-s**2)**2
    sbmodel[abs(s-1.)<1e-12] = 4./(15*a**2)

    return sbmodel/(2.*pi)


def jaffe_sb(rvals,a):
    """
    Calculate the projected Jaffe surface brightness distribution.

    rvals - locations to evaluate the model
    a     - scale radius
    """
    from scipy import log,arccos

    s0 = np.atleast_1d(abs(rvals)/a)
    sbmodel = s0*0.
    s = s0[s0<1.]
    sbmodel[s0<1.] = ((2.-s**2)*(1.-s**2)**-1.5)*log(1./s + (s**-2. - 1.)**0.5)
    sbmodel[s0<1.] = 2.*((1-s**2)**-1 - sbmodel[s0<1.])
    s = s0[s0>1.]
    sbmodel[s0>1.] = ((s**2 - 2)*(s**2 - 1.)**-1.5)*arccos(1./s)
    sbmodel[s0>1.] = -2.*((s**2 - 1)**-1 + sbmodel[s0>1.])
    sbmodel[abs(s0-1)<1e-12] = -8./3
    s = s0.copy()
    sbmodel += pi/s

    return sbmodel/(4.*pi*a**2)

def hernquist(r,a,proj=False):
    if proj:
        return hernquist_sb(r,a)
    return a/(2.*pi*r*(r+a)**3)

def jaffe(r,a,proj=False):
    if proj:
        return jaffe_sb(r,a)
    return a/(4.*pi*r**2*(r+a)**2)

def tPIEMD(r,lp_pars,proj=False):
    c, t = lp_pars
    if proj:
        return (pi/(c**2+r**2)**0.5 - pi/(t**2+r**2)**0.5)/(2*pi**2*(t-c))
    return (1./(r**2 + c**2) - 1./(r**2 + t**2))/(2*pi**2*(t-c))

def doubletPIEMD(r,lp_pars,proj=False):
    a1, c1, t1, a2, c2, t2 = lp_pars

    if proj:
        return a1*tPIEMD(r,[c1,t1],proj=True) + a2*tPIEMD(r,[c2,t2],proj=True)
    return a1*tPIEMD(r,[c1,t1]) + a2*tPIEMD(r,[c2,t2])

def doubleJaffe(r,lp_pars,proj=False):
    a1, r1, a2, r2 = lp_pars

    if proj:
        return a1*jaffe(r,r1,proj=True) + a2*jaffe(r,r2,proj=True)
    return a1*jaffe(r,r1) + a2*jaffe(r,r2)

def doubleHer(r, lp_pars, proj=False):
    a1, r1, a2, r2 = lp_pars

    if proj:
        return a1*hernquist(r,r1,proj=True) + a2*hernquist(r,r2,proj=True)
    return a1*hernquist(r,r1) + a2*hernquist(r,r2)

def sersic_bfunc(nser):
    # b(n), from Ciotti & Bertin 1999, A&A, 352, 447
    return 2*nser - 1./3. + 4/405./nser + 46/25515/nser**2

def sersic_sb(rvals, lp_pars):
    # Sersic surface brightness profile, normalized to unit luminosity
    reff, nser = lp_pars
    L = reff**2*2*np.pi*nser/sersic_bfunc(nser)**(2*nser)*gfunc(2*nser)
    return np.exp(-sersic_bfunc(nser)*(rvals/reff)**(1./nser))/L

def deVaucouleurs(r,reff,proj=False):
    if proj:
        return sersic_sb(r, (reff, 4.))
    else:
        return splev(r/reff,deV_rho_spline)*reff**(-3)

def sersic_3d_spline(lp_pars, rmin=1e-3, rmax=1e3, nr_grid=1001):
    # spherically de-projected Sersic profile, evaluated on a radial grid.

    r_grid = np.logspace(np.log10(rmin), np.log10(rmax), nr_grid)

    reff, nser = lp_pars
    def rho(r): # stellar density
        deriv = lambda R: -sersic_bfunc(nser)/nser*(R/(reff))**(1/nser)/R*sersic_sb(R, lp_pars)
        return -1./np.pi*quad(lambda R: deriv(R)/(R**2 - r**2)**0.5, r, np.inf)[0]

    rho_grid = np.zeros(nr_grid)
    for i in range(nr_grid):
        rho_grid[i] = rho(r_grid[i])

    return splrep(r_grid, rho_grid)


def sersic(r,lp_pars,proj=False):
    
    reff, nser = lp_pars

    if proj:
        return sersic_sb(r, lp_pars)
    else:
        # spherically deprojected Sersic profile
        deriv = lambda R: -sersic_bfunc(nser)/nser*(R/(reff))**(1/nser)/R*sersic_sb(R, lp_pars)
        return -1./np.pi*quad(lambda R: deriv(R)/(R**2 - r**2)**0.5, r, np.inf)[0]


