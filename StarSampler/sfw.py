import numpy as np
import scipy.special as ss


# @param X: list of input position coordinates (i.e. [r]).
#        V: list of input velocity coordinates (i.e. [vr, vt]).
#	 model_param = [a,d,e, Ec, rlim, b, q, Jb, rho, rs, alpha, beta, gamma] 
#	 context = [P0, Plim]
#          P0: DM potential at the center of the galaxy; precompute using genphi0 function at r=0
#          Plim: DM potential at rlim; precompute using genphi0 function at r=rlim
#          Note that P0 and Plim is precomputed to speedup the potential function evaluation. 
#	   Potential is 0 at the center of the DM halo.
def sfw_fprob(X, V, model_param, context):

        r, = X
        vr, vt = V

        #note: rhos = 4Pi*G*rho, unit: [(km/s)^2 (1/kpc^2)]
        a,d,e, Ec, rlim, b0, b1, alp, q, Jb, rho, rs, alpha, beta, gamma = model_param
        P0, Plim = context

	G = 4302 # (kpc/m_sun) (km/s)^2 *10^9 -- 10^9 is to convert rho's unit to [m_sun/kpc^3]
	rhos = 4*np.pi*G*rho


	#by constrution alp needs to be positive when b1>b0, otherwise negative.	
	if b1>b0:
		alp = abs(alp)
	else:
		alp = -1*abs(alp)

        Ps = rhos * rs**2   #unit (km/s)**2  #Vmax=0.465*sqrt(Ps)
        Pr = genphi0(r, rhos, rs, alpha, beta, gamma) - P0

        J  = abs(r * vt)              #J = v * r*sin(theta)
        E  = (vt*vt + vr*vr)/2.0 + Pr #E =  v*v/2 + Pr

        Ec   = Ec * Ps
        xlim = rlim / rs 
        Jb   = Jb * rs * (Ps**0.5) 

	gJ = ( (J/Jb)**(b0/alp) + (J/Jb)**(b1/alp) )**alp

        N  = 1.0*10**3

	E = E * (E<Plim) * (E>0)
	hE = np.nan_to_num( N*(E**a) * ((E**q + Ec**q)**(d/q)) * ((Plim - E)**e) )


        return hE * gJ * (4*np.pi*r*r * 2*np.pi*vt)     


def sampler_input(model_param):
	a,d,e, Ec, rlim, b0, b1, alp, q, Jb, rho, rs, alpha, beta, gamma = model_param
	G = 4302 # (kpc/m_sun) (km/s)^2 *10^9 -- 10^9 is to convert rho's unit to [m_sun/kpc^3]
        rhos = 4*np.pi*G*rho

        rlim = [0, model_param[4]]

        P0   = genphi0(1e-10, rhos, rs, alpha, beta, gamma)
        Plim = genphi0(rlim[1], rhos, rs, alpha, beta, gamma) - P0
	vmax = (2 * abs(Plim - 0))**0.5

	nX = 1
	nV = 2
        context = [P0, Plim]
        vlim = [0, vmax]

	return nX, nV, context, rlim, vlim



##----------------------------------------------------
#define the general potential
#note: rhos = 4Pi*G*rho*10^9, unit: [(km/s)^2 (1/kpc^2)]
def genphi0(r, rhos, rs, alpha, beta, gamma):

        if (alpha==beta and beta == gamma and gamma==99):
                return burkert_phi(r, rhos, rs)

        x = r/rs #+ 1e-10
        Ps = rhos * rs**2

        x0 = 10**-15
	alpha, beta, gamma = alpha*1.0, beta *1.0, gamma*1.0

        p1a = ss.hyp2f1((3.-gamma)/alpha, (beta*1.-gamma)/alpha, (3.+alpha-gamma)/alpha, -x0**alpha)
        p1b = ss.hyp2f1((3.-gamma)/alpha, (beta-gamma)/alpha, (3+alpha-gamma)/alpha,  -x**alpha)
        I1  = ( x0**(3-gamma) * p1a - x**(3-gamma) * p1b ) / (x * (gamma - 3.))
        #I1 = (0 - x**(3-gamma) * p1b ) / (x * (gamma - 3)) #to save calculation time 

        p2  = ss.hyp2f1( (-2.+beta)/alpha,(beta-gamma*1.)/alpha,(-2.+alpha+beta)/alpha, -x**(-alpha))
        I2  = x**(2.-beta) * p2 / (beta -2.)
        ans1 = Ps * ( 1 - (I1 + I2) )

        return ans1

#-----------burkert potential profile
def burkert_phi(r, rhos, rb):
        x = r/rb
        Ps = rhos * rb**2
        P = (1-1./x) * (np.log(x*x+1)/4.) + (1+1./x)*0.5*(np.arctan(x) - np.log(x+1))
        return Ps*P



