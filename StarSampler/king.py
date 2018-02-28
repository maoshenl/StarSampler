import numpy as np
import random as rand
from scipy import integrate
from scipy.interpolate import PchipInterpolator
from math import erf
from scipy import special

# @param X: a list of input position coordinates (i.e. [x = r/king_radius]).
#        V: a list of input velocity coordinates (i.e. [v]).
#	 model_param: [psi0, ratio0], the relative potential at center and its 
#	 	ratio with velocity dispersion square.
#	 context: [fP], the interpolated function of Phi/sigma^2 versus x.
def king_fprob(X, V, model_param, context):

	x, = X  #x=r/king_radius
	v, = V
	psi0, ratio0, = model_param
	fP, = context

        sigma2= (psi0/ratio0)
        E = fP(x) - v*v/(2.*sigma2)  #note that E = E/sigma^2

        rho1 = 1.

	E = E * (E>0)

	DF = rho1*(2*np.pi*sigma2)**(-1.5) * (np.exp(E) - 1)

        return DF* v*v*x*x *(4.*np.pi)**2


def sampler_input(model_param):
	psi0, ratio0, = model_param
        xarr, psi_sigma2, fP = getPsigma2(ratio0)

	nX = 1
	nV = 1
        context = [fP]
        rlim = [0, np.max(xarr)]
	vlim = [0 , (2.0*psi0)**0.5]

	return nX, nV, context, rlim, vlim



#solve the differential equation to get [Phi/sigma^2]
def getPsigma2(ratio0=3.0):

        rho0_rho1 = (np.exp(ratio0)*erf(ratio0**0.5) - ((4*ratio0/np.pi)**0.5)*(1+2*ratio0/3.))
        def deriv(u,x):    #r in kpc, x = r/r0 ,  r0 is king radius (0.5013 of entral brightness)
            u0 = u[0]
            u1 = u[1]
            if  u0<0:  #need to investigate negative values
                u0=0
		u1=0
            u1prime =(  (-u1*2.0/x) -
                        (9./(rho0_rho1)) * (np.exp(u0)*erf(u0**0.5)  - ((4*u0/np.pi)**0.5)*(1+2*u0/3.)) )
            u0prime = u1
            uprime  = np.array([u0prime,u1prime])

            return uprime

	# see Fig 4.9 of Binney&Tremaine (2008) for this guessed tidal radius.
	# because we want to evaluate the equation up-until the tidal radius, if not, retry 
	guessed_xmax = 2 + np.exp(.55*ratio0)
        x  = np.linspace(1e-8, guessed_xmax, 1e4)  
        uinitial = np.asarray([ratio0, 0])

        psi_sigma2 = integrate.odeint(deriv, uinitial, x, printmessg=0)[:,0] 
			#[:,1] contains the values of psi_prime

	x_trunc = x[psi_sigma2>0] #that's the tidal radius 
	xmax = np.max(x_trunc)
	while xmax >= guessed_xmax:
		guessed_xmax = guessed_xmax * 10
		x  = np.linspace(1e-8, guessed_xmax, ratio0*1e4)
		psi_sigma2 = integrate.odeint(deriv, uinitial, x, printmessg=0)[:,0]
		x_trunc = x[psi_sigma2>0]
	        C = np.max(x_trunc)
		print 'increase guessed_xmax by x10, retry. '
	psi_sigma2 = psi_sigma2[psi_sigma2>0]

	fP = PchipInterpolator(x_trunc, psi_sigma2, extrapolate=False)
	return x_trunc, psi_sigma2, fP
	


