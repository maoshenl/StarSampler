import numpy as np
import random as rand
from scipy import integrate
from scipy.interpolate import PchipInterpolator
from math import erf
from scipy import special




class King(object):
    def __init__(self, **model_param):
        '''
        Reading in the model parameters, and perform necessary calculations 
        to setup calculation of the model density function DF(X,V).

        It is required that this __init__( ) function contains:
            self.sampler_input = [self.nX, self.nV, self.Xlim, self.Vlim], where
                              nX (int)   : number of spatial variables.
                              nV (int)   : number of velocity variables.
                              Xlim (list): [min, max], range of the spatial variables
                              Vlim (list): [min, max], range of the velocity variables. 
        '''

        # Perform necessary calculations to setup King model, and use for DF calculation
        self.model_param = model_param
        sigma  = model_param['sigma']
        ratio0 = model_param['ratio0']
        rho0   = model_param['rho0']
        xarr, psi_sigma2, fP = getPsigma2(ratio0) #calculate king model potential

        sigma2 = sigma*sigma 
        psi0 = ratio0*sigma2
        G = 4.302*1e-6 #(kpc/m_sun) (km/s)^2 
        self.r0 = (9*sigma2/(4*np.pi*G*rho0))**.5 #king radius [kpc]

        self.param_list = [sigma, ratio0, rho0]
        self.model_param = model_param  #
        self.context = [fP]


        # Required user input regarding this specific model
        self.nX = 1
        self.nV = 1
        self.Xlim = [0, np.max(xarr) * self.r0 ]
        self.Vlim = [0 , (2.0*psi0)**0.5]

        self.sampler_input = [self.nX, self.nV, self.Xlim, self.Vlim]


    def DF(self, X, V):
        '''
        SFW probability function

        @param X: list of input position coordinates (i.e. [r]).
               V: list of input velocity coordinates (i.e. [v]).

        Results from __init__() needed for the DF(X,V) calculation:  
           a) model param_list = [a,d,e, Ec, rlim, b0, b1, alp, q, Jb, rho, rs, alpha, beta, gamma] 
           b) context = [fP], where
                     fP(x): potential function of king model, where x = r/r0, 
                            where r0 is the king radius

        Return: probability of coordinates in X, V
        '''
        
        r, = X
        v, = V
        sigma, ratio0, rho0 = self.param_list

        sigma2 = sigma*sigma
        fP, = self.context

        x = r/self.r0  #x=r/king_radius
        E = fP(x) - v*v/(2.*sigma2)  #note that E = E/sigma^2
        rho1 = 1. #constant is inrelevant for sampling

        E = E * (E>0)
        DF = rho1*(2*np.pi*sigma2)**(-1.5) * (np.exp(E) - 1)

        return DF* v*v*x*x *(4.*np.pi)**2



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
        


