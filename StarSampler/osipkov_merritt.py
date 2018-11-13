import numpy as np
import random as rand
import scipy.special as ss
import scipy.optimize
import warnings
import time
from scipy import integrate
from scipy.interpolate import PchipInterpolator


class OM(object):
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

        ra   = model_param['ra']
        rs_s = model_param['rs_s']
        al_s = model_param['al_s']
        be_s = model_param['be_s']
        ga_s = model_param['ga_s']
        rho  = model_param['rho']
        rs   = model_param['rs']
        alpha= model_param['alpha']
        beta = model_param['beta']
        gamma= model_param['gamma'] 

        G = 4.302*1e-6 # (kpc/m_sun) (km/s)^2 * 10^9
        self.rhos = 4*np.pi*G*rho
        DM_param = [self.rhos, rs, alpha, beta, gamma]
        self.param_list = [ra, rs_s, al_s, be_s, ga_s, rho, rs, alpha, beta, gamma]


        #-- num_rsteps = 1e5 should be enough of a table to approximate drho(r)/dPhi(r) 
        #-- num_Qsteps = 1000 should be enough to create G(Q) table to approximate f(Q)
        Qarr, dfG, rtrunc = GQ(self.param_list, num_rsteps = 1e5, num_Qsteps = 1000)
        r200 = getR200(DM_param)

        Pr0 = OMgenphi(1e-8, rtrunc, self.rhos, rs, alpha, beta, gamma)
        Plim0 = 0
        vmax = (2 * abs(Plim0 - Pr0))**0.5

        self.nX = 1
        self.nV = 2
        self.Xlim = [0, r200]
        self.Vlim = [0, vmax]
        self.context = [Qarr, dfG, rtrunc]
        self.model_param = model_param
   
        # Required output of __init__()
        self.sampler_input = [self.nX, self.nV, self.Xlim, self.Vlim]



    def DF(self, X, V):
        '''
        Osipkov-Merritt-Zhao probability function.

        @param X: list of input position coordinates (i.e. [r]).
               V: list of input velocity coordinates (i.e. [vr, vt]).

        Required from class initialization:  
               a) model parameter list = [ra, rs_s, al_s, be_s, ga_s, rho, rs, alpha, beta, gamma] 
               b) context = [Qarr, fQ, rtrunc], where
                      Qarr: Q array that fits G(Q) and f(Q). 
                      fQ: f(Q) funtion
                      rtrunc: the radius at which DM density is truncated, currently set at R200

        Return (array): probability of coordinates in X, V
        '''

        r, = X
        vr, vt = V
        Qarr, fQ, rtrunc = self.context        
        ra, rs_s, al_s, be_s, ga_s, rho, rs, alpha, beta, gamma = self.param_list
 
        Pr = 0 - OMgenphi(r, rtrunc, self.rhos, rs, alpha, beta, gamma) 
        Q  = Pr - ( vr*vr + (1+r*r/(ra*ra))*vt*vt )*0.5

        try:
            if (len(Q)>0):
                Qup = Q>np.max(Qarr)
                Qlo = Q<np.min(Qarr)
                p = fQ(Q)
                p[Qup] = 0
                p[Qlo] = 0
                p = p * (Q>0) # so that probability is zero when Q<=0
                
        except:
            p = fQ(Q) if ( Q > np.min(Qarr) and Q < np.max(Qarr) ) else 0        


        return p * (4*np.pi*r*r * 2*np.pi*vt) 




    def conditional_sample(self, samplesize, Phi_table_steps=1e5, GQ_table_steps=1000,
                proposal_steps = 1000, r_vr_vt=False):    
        '''
        Sampling the Osipkov-Merritt-Zhao model by conditional probability 
        (i.e. first sampling r from p(r), then Q from p(Q|r), etc.)

        @param samplesize (int): number of sample to draw from the DF(X, V).
               Phi_table_steps (int): number of points in radius (r) use to create 
                                potential Phi(r) and mass density rho(r) tables.
               GQ_table_steps (int): number of points in Q use to to create G(Q) tables.
               proposal_steps (int): number of steps in the proposal functions that's 
                                     used to sample r and Q.
               r_vr_vt (bool): if True convert obtained samples from (r, v_r, v_t) 
                               to (x,y,z, vx, vy, vz).

        Return (array): (r, v_r, v_t) or (x,y,z,vx,vy,vz) sample arrays, depends on r_vr_vt flag.
        '''

        ans = OM_sample(self.param_list, self.context, samplesize, Phi_table_steps, GQ_table_steps,
                proposal_steps, r_vr_vt)

        return ans




###------------------------------------------------------------------------------####
###                                Support Functions                             ####
###------------------------------------------------------------------------------####

#alpha beta gamma potential energy, density is truncated at rtrunc 
def OMgenphi(r, rtrunc, rhos, rs, alpha, beta, gamma):
        x0 = 10**-12
        xlim = (rtrunc + 0) / rs
        alpha = alpha*1.0 #in case alpha is int 
        beta  = beta *1.0
        gamma = gamma*1.0

        x = r/rs #+ 1e-10
        Ps = rhos * rs**2

        x2 = x
        try:
            x2[x2>xlim] = xlim        
        except:
            x2 = x2 if x2<xlim else xlim

        p2a = ss.hyp2f1( (2.-gamma)/alpha,(beta-gamma*1.)/alpha, (2.+alpha-gamma)/alpha, -x2**(alpha) )
        p2b = ss.hyp2f1( (2.-gamma)/alpha,(beta-gamma*1.)/alpha, (2.+alpha-gamma)/alpha, -xlim**(alpha) )
               I2  = (x2**(2-gamma) * p2a - xlim**(2-gamma) * p2b) / (gamma - 2)

        p1a = ss.hyp2f1((3.-gamma)/alpha, (beta*1.-gamma)/alpha, (3.+alpha-gamma)/alpha, -x0**alpha)
        p1b = ss.hyp2f1((3.-gamma)/alpha, (beta-gamma)/alpha, (3+alpha-gamma)/alpha,  -x2**alpha)
        I1  = ( x0**(3-gamma) * p1a - x2**(3-gamma) * p1b ) / ((r/rs) * (gamma - 3.))
        #I1 = (0 - x**(3-gamma) * p1b ) / (x * (gamma - 3)) #to save time ignore p1a..(for gamma<3 or so)

        ans = Ps * ( 0 - (I1 + I2) ) #

        return ans


def rho_Q(rarr, ra, rs_s, al_s, be_s, ga_s):
        rhos_s = 1.0
        rhoQ1 = ( (1+rarr*rarr/(ra*ra))
                  * rhos_s * ((rarr/rs_s)**-ga_s) * (1+(rarr/rs_s)**al_s)**((ga_s-be_s)/al_s) )
        return rhoQ1

# calculate f(Q) function
def GQ(model_param, num_rsteps = 1e5, num_Qsteps = 1000):
        time_ini = time.time()
        print 'Calculating f(Q) function... '

        ra, rs_s, al_s, be_s, ga_s, rho, rs, alpha, beta, gamma = model_param
        G = 4.302 * 1e-6 # (kpc/m_sun) (km/s)^2
        rhos = 4*np.pi*G*rho

        DM_param = [rhos, rs, alpha, beta, gamma]

        rhos_s = 1.0
        al_s, be_s, ga_s = al_s*1.0, be_s*1.0, ga_s*1.0

        nsteps = 1e5
        r200   = getR200(DM_param)
        rmax   = r200*1000000
        rtrunc = r200*1 #Dark matter density cut off radius
        Plim = 0  

        #---------------we want drho/dP over large range of rarr, so use rmax----------------
        t0 = time.time()
        rarr0 = np.linspace(1e-8, rmax*1, num_rsteps*.5)
        rarr1 = np.logspace(-5, np.log10(rmax)-0, num_rsteps*.5)
        rarr2 = np.logspace(-8, np.log10(rmax)-6, num_rsteps*.5)
        rarr = np.unique( np.hstack((rarr0, rarr1, rarr2)) )
        rarr = rarr[np.argsort(rarr)]
        Parr = -1*(OMgenphi(rarr, rtrunc, rhos, rs, alpha, beta, gamma)) 
        #print 'calculate potential time: ', time.time()-t0 

        # -------- interpolate between rhoQ(r) and Phi(r) -------------------------
        rhoQ = rho_Q(rarr, ra, rs_s, al_s, be_s, ga_s)

        rhoQ_sorted = rhoQ[np.argsort(Parr)]
        Parr_sorted = Parr[np.argsort(Parr)]
        Parr_sorted, Pindx = np.unique(Parr_sorted, return_index=True)
        rhoQ_sorted = rhoQ_sorted[Pindx]

        t0 = time.time()
        frhoQ  = PchipInterpolator(Parr_sorted, rhoQ_sorted, extrapolate=False)
        dfrhoQ = frhoQ.derivative()
        #print "interpolate rhoQ and relative potential time:  ",  time.time()-t0

        #-------- calculate G(Q) -------------------------------------------------------
        def G_integrand(u, Q):
            phi = Q-u*u
            return -2 * dfrhoQ(phi) 


        t0 = time.time()
        rarr0 = np.logspace(-8, np.log10(r200)+1, int(num_Qsteps*.35))
        Qarr0 = -1*(OMgenphi(rarr0 , rtrunc, rhos, rs, alpha, beta, gamma) - Plim )
        Qarr1 = np.linspace(0, max(Parr_sorted)*1, int(num_Qsteps*.65))
        Qarr = np.hstack((Qarr1, Qarr0))
        Qarr = np.unique(Qarr)
        Qarr = Qarr[np.argsort(Qarr)]

        Garr = [integrate.quad(G_integrand, Q**.5, 0, args=(Q,), full_output=1, 
                epsabs=0, epsrel=1.49e-05)[0] for Q in Qarr]

        Garr = np.nan_to_num( np.array(Garr) )
        #print 'G(Q) integrate time: ', time.time()-t0

        #------------ interpolate Qarr and Garr to get f(Q) ----------------------
        Garr = Garr/(np.pi*np.pi* 2**1.5)
        GQ = PchipInterpolator(Qarr, Garr, extrapolate=False)
        fQ = GQ.derivative() 
        fQarr = fQ(Qarr)
        
        print '    Finish calculating f(Q). Time used (sec): ', time.time()-time_ini

        #-------------------check if f(Q) is negative --> unphysical--------------
        numQ = 20000
        Qtest = np.linspace(0, max(Qarr)*1, numQ)
        fQtest = fQ(Qtest)
        num_neg_fQ = sum(fQtest<0)
        neg_Q_fraction = num_neg_fQ / (numQ*1.)
        if neg_Q_fraction >= 0.0001: #len(neg_fQ) > len(Qarr)*.01:
                print 'This model could be unphysical! please double check'
                print 'model_param = ', model_param
        #--------------------end check ----------------------------------

        return Qarr, fQ, rtrunc


#calculate r200 for a given dark matter parameters
def getR200(DM_param):
        rhos, rs, al, be, ga, = DM_param

        G = 4.302*10**-6 #kpc/M_sun * (km/s)^2
        H = .072 #km/s / kpc
        rho_c = 3*H*H/(8*np.pi*G)

        def f2(r):
            x0 = 1e-10
            auxx = r/rs
            p1_a = ss.hyp2f1((3.-ga)/al, (be-ga)/al, (3+al-ga)/al, -x0**al)
            p1_b = ss.hyp2f1((3.-ga)/al,(be-ga)/al,(3.+al-ga)/al, -auxx**al)
            #note: rhos = 4Pi*G*rhos = ((21/.465)(1/rs))^2 = 4229.2 (km/s)^2 (1/kpc^2)
            Mr_DM = (rhos/G) * ( x0**(3.-ga) * p1_a - auxx**(3.-ga) * p1_b ) / ((ga - 3.))
            return Mr_DM*rs**3

        rarr = np.linspace(rs, rs*10000, 1000000)
        Mc = (4*np.pi/3.0) * rarr**3 * rho_c * 200.
        Mr = f2(rarr)
        min_indx = np.argmin(abs(Mr-Mc))

        try:
                rarr2 = np.linspace(rarr[min_indx-5], rarr[min_indx+5], 1000)
                Mc2 = (4*np.pi/3.0) * rarr2**3 * rho_c * 200.
                Mr2 = f2(rarr2)
                min_indx2 = np.argmin(abs(Mr2-Mc2))
                r200 = rarr2[min_indx2]
        except:
                r200 = rarr[min_indx]

        return r200 * 1



#-------Following functions are for sampling based on marginalized probability--------

#-- num_rsteps = 1e5 should be enough of a table to approximate drho(r)/dPhi(r) 
#-- num_Qsteps = 1000 should be enough to create G(Q) table to approximate f(Q)
def OM_sample(model_param, context, samplesize, Phi_table_steps=1e5, GQ_table_steps=1000, 
        proposal_steps = 1000, r_vr_vt=False):

        samplesize, Phi_table_steps, GQ_table_steps, proposal_steps = \
            int(samplesize), int(Phi_table_steps), int(GQ_table_steps), int(proposal_steps)

        ra, rs_s, al_s, be_s, ga_s, rho, rs, alpha, beta, gamma = model_param
        G = 4.302*1e-6 # (kpc/m_sun) (km/s)^2
        rhos = 4*np.pi*G*rho
        #DM_param = [rhos, rs, alpha, beta, gamma]

        Qarr, fQ, rtrunc = context  #GQ(model_param,  
        #        num_rsteps=Phi_table_steps, num_Qsteps=GQ_table_steps)

        rlim = rtrunc #rtrunc = r200 
        Plim = 0 #Plim0 

        t0 = time.time()
        Qarr0 = np.linspace(0, max(Qarr)*1, proposal_steps)
        Qprob_arr0 = fQ(Qarr0) 
        Qprob_arr0 = Qprob_arr0*(Qprob_arr0>0)
        dQ = Qarr0[1]-Qarr0[0]

        #------------------------------- Sample r ---------------------------------

        #--- build step function of the probability(r) ---
        rarr = np.linspace(1e-10,rlim, proposal_steps)
        rprob_arr = rprobability(model_param, rarr)
        dr = rarr[1]-rarr[0]
        rprob_proposal = []
        for i in xrange(len(rarr)-1):
                rprob_proposal.append( max(rprob_arr[i], rprob_arr[i+1]) )

        r_max, fr_max = rprob_max(model_param)
        rmax_index = sum(rarr<r_max)
        #print 'max_index: ', rmax_index, r_max, fr_max, max(rprob_proposal), len(rprob_proposal)
        rprob_proposal = np.array(rprob_proposal)

        if (rmax_index>0 and (rmax_index-1 < len(rprob_proposal))):
                rprob_proposal[rmax_index-1] = fr_max * 1.05

        norm_rprob_proposal = rprob_proposal/np.sum(rprob_proposal)
        #------------------------------------------------


        Pr0 = OMgenphi(1e-8, rtrunc, rhos, rs, alpha, beta, gamma)
        vmax = (2 * abs(0 - Pr0))**0.5

        N = 0
        r_list  = []
        vr_list = []
        vt_list = []
        if (True): 
            j=0
            computeN = 0
            acceptN = 0
            r_accept = []
            num  = samplesize
            auxN = samplesize
            eff = 1.0        
            while(True):
                anxN = int(auxN/eff)
                if (auxN > 1e7): # to limit the memory usage
                        auxN = int(1e7)

                #sampling r from proposal
                rindex = np.random.choice(np.arange(len(rprob_proposal)),size=auxN,p=norm_rprob_proposal)
                ux = rarr[rindex] + np.random.rand(auxN)*dr
                fproposal = rprob_proposal[rindex]
                
                uf = rprobability(model_param, ux)
                u = np.random.rand(auxN)
                accept_index = ( (uf/fproposal)>u )
                r_accept.append( ux[accept_index] )
        
                acceptN += np.sum(accept_index)
                computeN += auxN
                eff = acceptN/(computeN*1.0)
                eff = eff if eff>0 else .01                
                #print 'sample r', j, computeN, acceptN
                j = j+1
                if (acceptN >= samplesize):
                        break
                auxN = samplesize - acceptN

            rsamples = np.hstack(r_accept)[0:samplesize]
            #-----------------DONE sampling r--------------

            #for each r, sample Q        
            j =0
            null_count = 0 
            vr_samples = []
            vt_samples = []
            r_samples = []
            for ri in rsamples:

                #----------- building probability(Q) proposal ----------------------
                Pri = -(OMgenphi(ri, rtrunc, rhos, rs, alpha, beta, gamma) )
                Qmax = Pri
                accept = False
                num_guess=1

                Qmax_index = sum(Qarr0<Qmax)
                Qarr = Qarr0[:(Qmax_index+1)]
                Qprob_arr = Qprob_arr0[:(Qmax_index+1)] * (2*abs(Pri-Qarr))**.5 * (Pri>Qarr) 
                Qprob_proposal = np.array( [max(Qprob_arr[i], Qprob_arr[i+1]) 
                                for i in xrange(len(Qarr)-1)] )

                Q_max, fQ_max = Qprob_max(fQ, Pri)
                Qmax_index = sum(Qarr<Q_max) 

                #ONLY if Q_max is within the limit of Qarr then..
                if (Qmax_index>0 and (Qmax_index-1 < len(Qprob_proposal)) 
                    and fQ_max>Qprob_proposal[0] and fQ_max>Qprob_proposal[-1]):
                        Qprob_proposal[Qmax_index-1] = fQ_max * 1.05
                norm_Qprob_proposal = Qprob_proposal/np.sum(Qprob_proposal)
                #--------- Done building probability(Q) proposal---------------------                
        
                while(not accept):
                        #sampling from probability(Q) proposal
                        num_guess = num_guess*2 if num_guess<1e6 else 1e6
                        Qindex = np.random.choice(np.arange(len(Qprob_proposal)),
                                                  size=num_guess,p=norm_Qprob_proposal)
                        ux = Qarr[Qindex] + np.random.rand(num_guess)*dQ
                        fproposal = Qprob_proposal[Qindex]
        
                        uf = fQ(ux) * (2*(Pri-ux)*(Pri>ux))**.5 
                        u = np.random.rand(num_guess)
                        accept_ux = ux[((uf/fproposal)>u)] 

                        if len(accept_ux)>0:
                               for uxi in accept_ux:
                                Qi = uxi 

                                vrmax = (2*max(Pri-Qi,0))**.5
                                vri = rand.random() * vrmax                                
                                T2 = vri*vri

                                T1 = (Pri-Qi)*2
                                if T1>T2 :
                                        vti =( (T1 - T2)/(1+ri*ri/(ra*ra)) )**.5
                                        vr_samples.append(vri)
                                        vt_samples.append(vti)
                                        r_samples.append(ri)
                                        accept = True
                                        j = j+1
                                        if j%100==0:
                                            print "samples accepted: ", j #, num_guess 
                                        break
                                else:
                                        null_count = null_count+ 1                                        

        samplearr = np.vstack((r_samples,vr_samples,vt_samples))
        samplearr = np.swapaxes(samplearr,0,1)
        print 'OM sample time (sec): ', time.time()-t0 #, null_count

        if r_vr_vt:
                return r_vr_vt_complete(samplearr)

        return samplearr
        

#stellar density
def rprobability(model_param, r):
        rhos_s = 1
        ra, rs_s, al_s, be_s, ga_s, rho, rs, alpha, beta, gamma = model_param

        rprob = (r*r)*rhos_s * ((r/rs_s)**-ga_s) * (1+(r/rs_s)**al_s)**((ga_s-be_s)/al_s)         
        return rprob


def rprob_max(model_param):
        rhos_s = 1
        ra, rs_s, al_s, be_s, ga_s, rho, rs, alpha, beta, gamma = model_param

        def rfun(r):
                rprob = (r*r)*rhos_s * ((r/rs_s)**-ga_s) * (1+(r/rs_s)**al_s)**((ga_s-be_s)/al_s)
                return rprob 

        optvar = scipy.optimize.fmin(lambda (r1): -rfun(r1), (rs_s), disp=False)
        fmax = rfun(optvar) 
        return optvar, fmax


def Qprob_max(dfGQarr1, Pri):
        def Qfun(Q):
            if Q>Pri:
                return 0
            else:
                result = dfGQarr1(Q) * (2*(Pri-Q))**.5 
                return -result
        x1 = 0
        x2 = Pri
        optvar = scipy.optimize.fmin(lambda (Q1): Qfun(Q1), (Pri*.9), disp=False)
        #optvar = scipy.optimize.fminbound(Qfun, x1,x2, disp=False)
        fmax = -Qfun(optvar)
        return optvar, fmax


def r_vr_vt_complete(samplelist):
        if len(samplelist)>=1:
                samplearr = np.asarray(samplelist)
                rarr  = samplearr[:,0]
                vrarr = samplearr[:,1]
                vtarr = samplearr[:,2]

        else:   #in case it's empty..
                print "ERROR: Empty sampleset"

        #tranform to the dwarf-galaxy-centered cartesian coordinate
        r  = rarr
        v  = (vrarr*vrarr + vtarr*vtarr )**0.5

        rsize = len(r)
        ru    = np.random.random_sample( rsize )
        theta = np.arccos(1-2*ru) #inverse sampling the distribution for theta
        phi   = np.random.random_sample( rsize ) * 2.0 * np.pi

        vsign = np.sign( np.random.random_sample( rsize ) - 0.5 )
        vphi  = np.random.random_sample( rsize ) * 2.0 * np.pi

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        vz2 = vsign * vrarr
        vx2 = vtarr * np.cos(vphi)
        vy2 = vtarr * np.sin(vphi)

        #passive rotation, using rotation matrix 
        #to rotate the zhat of calculate velocity into the zhat of the spatial coordinate
        vx = np.cos(theta)*np.cos(phi)*vx2 - np.sin(phi)*vy2 + np.sin(theta)*np.cos(phi)*vz2
        vy = np.cos(theta)*np.sin(phi)*vx2 + np.cos(phi)*vy2 + np.sin(theta)*np.sin(phi)*vz2
        vz = -np.sin(theta)*vx2 + np.cos(theta)*vz2
        return x, y, z, vx, vy, vz        




