from scipy.interpolate import splrep,splev,splint
from scipy import ndimage,integrate
import numpy
from math import pi
import tracer_profiles as profiles


def radialConvolve(r,f,sigma,fk=100,fr=1):
    from scipy.special import j0
    import special_functions as sf
    #mod = splrep(r,f,s=0,k=1)
    #norm = splint(r[0],r[-1],mod)
    r0 = r.copy()
    f0 = f.copy()
    r = r/sigma
    sigma = 1.

    kmin = numpy.log10(r.max())*-1
    kmax = numpy.log10(r.min())*-1
    k = numpy.logspace(kmin,kmax,r.size*fr)
    a = k*0.
    for i in range(k.size):
        bessel = j0(r*k[i])
        A = splrep(r,r*bessel*f,s=0,k=1)
        a[i] = splint(0.,r[-1],A)

    a0 = (2.*pi*sigma**2)**-0.5
    b = a0*sigma*numpy.exp(-0.5*k**2*sigma**2)

    ab = a*b
    mod = splrep(k,ab,s=0,k=1)
    k = numpy.logspace(kmin,kmax,r.size*fk)
    ab = splev(k,mod)
    result = r*0.
    for i in range(r.size):
        bessel = j0(k*r[i])
        mod = splrep(k,k*bessel*ab,s=0,k=1)
        result[i] = 2*pi*splint(0.,k[-1],mod)

    return result


def Isigma2(R,r,M,light_profile,lp_args=None):
    #if type(R)==type(1.):
    #    R = numpy.asarray([R])
    R = numpy.atleast_1d(R)

    light = light_profile(r,lp_args)

    model = splrep(r,M*light/r**2,k=3,s=0) #this is part of the integrand
    result = R*0.
    for i in range(R.size):
        reval = numpy.logspace(numpy.log10(R[i]),numpy.log10(r[-1]),301)
        reval[0] = R[i] # Avoid sqrt(-epsilon)
        Mlight = splev(reval,model)
#	if ((reval**2).min() - R[i]**2)<0:
#	    print reval.min(),R[i]
#	    print 'porco dio!'

	eps = 10**(-8)
        integrand = Mlight*(reval**2-R[i]**2+eps)**0.5
        mod = splrep(reval,integrand,k=3,s=0)
        result[i] = 2.*splint(R[i],reval[-1],mod)
    return result


def Isigma2Beta(R,r,M,light_profile,lp_args,beta):
    from scipy.special import gamma,betainc,beta as B
    R = numpy.atleast_1d(R)
    light = light_profile(r,lp_args)

    model = splrep(r,M*light,k=3,s=0)
    t1 = 0.5*(1.5-beta)*(pi**0.5)*gamma(beta-0.5)/gamma(beta)
    result = R*0.
    for i in range(R.size):
        reval = numpy.logspace(numpy.log10(R[i]),numpy.log10(r[-1]),301)
        reval[0] = R[i] # Avoid sqrt(-epsilon)
        Mlight = splev(reval,model)
        u = reval/R[i]
        K = (1-u**-2)**0.5/(1.-2*beta) + t1*(u**(2*beta-1.))*(1.-betainc(beta+0.5,0.5,u**-2))
        mod = splrep(reval,K*Mlight/reval,k=3,s=0)
        result[i] = 2.*splint(R[i],reval[-1],mod)
    return result


def Isigma2OM(R,r,M,light_profile,lp_args,ra):
    #calculates the brightness weighted velocity dispersion (squared) for a Osipkov-Merrit anisotropy model
    if type(R)==type(1.):
        R = numpy.asarray([R])
    light = light_profile(r,lp_args)
    a = lp_args

    model = splrep(r,M*light,k=3,s=0)
    result = R*0.
    for i in range(R.size):
        reval = numpy.logspace(numpy.log10(R[i]),numpy.log10(r[-1]),301)
        reval[0] = R[i] # Avoid sqrt(-epsilon)
        Mlight = splev(reval,model)
        u = reval/R[i]
        ua = ra/R[i]
        K = (ua**2+0.5)/(ua**2+1.)**1.5*(u**2+ua**2)/u*numpy.arctan(numpy.sqrt((u**2-1)/(ua**2+1))) - 0.5/(ua**2+1)*numpy.sqrt(1-1/u**2)
        mod = splrep(reval,K*Mlight/reval,k=3,s=0)
        result[i] = 2.*splint(R[i],reval[-1],mod)
    return result




import cPickle
import vdmodel_2013
path = vdmodel_2013.__path__[0]
#sbSeeingModel = cPickle.load(open('%s/sbSeeing.model'%path))

def ObsSigma2(aperture,r,M,seeing,light_profile,lp_args,anisotropy,anis_par,inner=None,limit=None,multi=False,beta=0.,doConv=False):

    if type(aperture)==list or type(aperture)==tuple:
        if len(aperture)==4:
            x1,x2,y1,y2 = aperture
    #        R = ((x1-x2)**2+(y1-y2)**2)**0.5
            R = (x2**2+y2**2)**0.5
        else:
            dx,dy = aperture
            x1,x2,y1,y2 = dx/-2.,dx/2.,dy/-2.,dy/2.
            R = ((x1-x2)**2+(y1-y2)**2)**0.5
    else:
        R = aperture

    if limit is None:
        limit = r[-1]
    if inner is None:
        inner = r[0]
    Rvals = r[r<=limit]
    Rvals = r[:-1]

    if anisotropy is None:
        sbSigma = Isigma2(Rvals,r,M,light_profile,lp_args)
    elif anisotropy=='beta':
        sbSigma = Isigma2Beta(Rvals,r,M,light_profile,lp_args,anis_par)
    elif anisotropy=='OM':
        sbSigma = Isigma2OM(Rvals,r,M,light_profile,lp_args,anis_par)

    index = numpy.where(R<=Rvals)[0][0]+1
    import time
    if seeing is not None:
        seeing /= 2.355

        sigma = radialConvolve(Rvals,sbSigma,seeing,1.)
        sb = light_profile(Rvals,lp_args,proj=True)
        if doConv is True:
            sbmodel = radialConvolve(Rvals,sb,seeing,1.)
        else:
            eval = numpy.empty((Rvals.size,3))
            eval[:,0] = Rvals.copy()
            eval[:,1] = Rvals*0. + lp_args
            eval[:,2] = Rvals*0. + (2.355*seeing/lp_args)
            sbmodel = sbSeeingModel.eval(eval)

    else:
        sbmodel = light_profile(Rvals,lp_args,proj=True)
        sigma = sbSigma.copy()


    if type(aperture)==list or type(aperture)==tuple:
        #rectangular aperture
            
        r1 = Rvals[:index].copy()
        sigma = sigma[:index]
        sbmodel = sbmodel[:index]

        radsigma = splrep(r1,sigma)
        radsbmodel = splrep(r1,sbmodel)
        
        sigmaintegrand = lambda x,y: splev((x**2+y**2)**0.5,radsigma)
        sbintegrand = lambda x,y: splev((x**2+y**2)**0.5,radsbmodel)

        integral2 = integrate.dblquad(sigmaintegrand,x1,x2,lambda y:y1,lambda y:y2)[0]/integrate.dblquad(sbintegrand,x1,x2,lambda y:y1,lambda y:y2)[0]
        return R,integral2

    else:
        r1 = Rvals[:index].copy()
        sigma = sigma[:index]
        sbmodel = sbmodel[:index]
        sigmod = splrep(r1,r1*sigma)
        sbmodel = splrep(r1,r1*sbmodel)
        if multi is False:
            return R,splint(inner,R,sigmod)/splint(inner,R,sbmodel)
        rout = Rvals[Rvals>inner]
        vd = rout*0.
        vd2 = rout*0.
        for i in range(rout.size):
            vd[i] = splint(inner,rout[i],sigmod)/splint(inner,rout[i],sbmodel)
        return rout,vd


def sigma2(Mfunc, aperture, lp_pars, light_profile=profiles.deVaucouleurs, seeing=None, anisotropy=None, anis_par=None, reval0=None):
    """
    returns sigma^2/G for a generic mass distribution specified by Mfunc. Has same units as Mfunc/aperture.

    Parameters
    ----------
    Mfunc : function or tuple. 
        3D mass distribution of the galaxy. If function, must take an array of radii and return the mass enclosed within each shell of the corresponding radius. If tuple, the first element must be an array of radii, the second element must be an array of masses enclosed in shells of radius matching the fist element
    aperture : float, list or tuple.
        Aperture within which velocity dispersion is calculated. If float, the aperture is a circle centered on the galaxy center. If list or tuple with len==2, then the two elements are the width and height of a rectangular aperture centered on the galaxy center. If list or tuple with len==4, then the four elements are the coordinates of a rectangle in an arbitrary position.
    lp_pars : float or iterable.
        Arguments used by the tracer distribution function, defining the parameters of the light profile (e.g. half-light radius, Sersic index, etc.)
    light_profile : function
        Density profile of the tracers. Takes lp_pars as second argument.
    seeing : float.
        Seeing FWHM
    anisotropy : str.
        Family of anisotropy profile models; options: [None, 'Beta', 'OM'], corresponding to isotropic, constant anisotropy (beta=cnst) and Osipkov-Merritt anisotropy profile.
    anis_par : float.
        Anisotropy parameter. If anisotropy=='beta', then this is a constant anis_par = beta = 1 - sigma_theta^2/sigma_r^2 . If anisotropy=='OM', this is the scale radius of an Osipkov-Merritt anisotropy profile: beta = r^2/(r^2 + anis_par^2)
    reval0 : float.
        Physical scale of the galaxy. Default: None. If None, it is set to the half-light radius of the stellar profile. This quantity is used to set the radial range of the integration of the Jeans equation. If Mfunc is a tuple, it is ignored.

    Returns
    -------
    s2 : float
        sigma^2/G in units of Mfunc/aperture. 
    """

    light = light_profile

    if type(Mfunc) == tuple:
        r_eval,M = Mfunc
    else:
        if reval0 is None:
            if type(lp_pars) is list or type(lp_pars) is tuple:
                reval0 = lp_pars[0]
            else:
                reval0 = lp_pars
        logreval0 = numpy.log10(reval0)
        r_eval = numpy.logspace(logreval0-3,logreval0+3,400)
        M = Mfunc(r_eval)
    s2 = ObsSigma2(aperture,r_eval,M,seeing,light,lp_pars,anisotropy,anis_par,doConv=True)[1]
    return s2


