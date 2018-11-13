import numpy as np
import random as rand
import scipy.optimize
import scipy.special as ss
import time



def impt_sample(model_class, steps, resample_factor, samplesize, 
                     replace=True, r_vr_vt=False, r_v=False, filename=None):
    '''
    Sampling Importance Resampling (SIR) method, or importance sampling.
    @param model_class (class): A model class object that contains the probability function DF(X,V),
                        and the sampler_input list = [nX, nV, Xlim, Vlim], 
                        where nX (int): number of spatial variables.
                              nV (int) : number of velocity variables.
                              Xlim (list): [min, max], range of the spatial variables
                              Vlim (list): [min, max], range of the velocity variables.

           steps (int): Number of steps in the proposal function. (A uniform function has a step of 1).

           resample_factor (int): a multiple of samplesize that sets the total proposal points.
                                  (i.e. number of proposal points = resample_factor * samplesize).

           samplesize (int): number of samples to draw from DF(X,V)

           replace (bool): if True proposal points will be drawn with replacement, otherwise
                           without replacement. (replace=True is recommended.) 
    
           r_vr_vt (bool): if True transform (r, vr, vt) --> (x,y,z, vx,vy,vz), 
                              where vr, vt are radial and tangential velocities.

           r_v (bool): if true transform (r, v) --> (x,y,z, vx,vy,vz).

           filename (string): if given a filename, the output will be saved to a text file 
                              the given file name.

    Return: (array) sample drawn of the phase variables in (X,V). 
            Return [x,y,z, vx,vy,vz] arrays if either r_vr_vt or r_v is true.
    '''


    nX, nV, Xlim, Vlim = model_class.sampler_input
    nA = 0

    if nX+nV > 6:
        print 'We only support a maximum of 6 variables'
        return
    if (r_vr_vt and r_v)==True:
        print 'You cannot have both r_vr_vt AND r_v set to True'
        return
    if (Xlim[1] <= Xlim[0] or Vlim[1] <= Vlim[0]):
        print 'ERROR: rmax <= rmin or vmax<=vmin, please double check the sample limits'


    ans = imptsample(model_class.DF, Xlim, Vlim, nX, nV, nA,
                [], [], steps, samplesize, resample_factor, replace,
                r_vr_vt, r_v, z_vr_vt=False)

    if filename != None:
        if (r_vr_vt or r_v ):
            np.savetxt(filename, np.c_[ans])
        else:
            np.savetxt(filename, np.c_[np.transpose(ans)])

    return ans




def rejection_sample(model_class, samplesize, r_vr_vt=False, r_v=False, filename=None, brute=True):
    '''
    Rejection sampling method.
    @param model_class (class): A model class object that contains the probability function DF(X,V),
                        and the sampler_input list = [nX, nV, Xlim, Vlim], 
                        where nX (int): number of spatial variables.
                              nV (int) : number of velocity variables.
                              Xlim (list): [min, max], range of the spatial variables
                              Vlim (list): [min, max], range of the velocity variables.
           samplesize (int):  number of samples to draw from DF(X,V)
           r_vr_vt    (bool): if true transform (r, vr, vt) --> (x,y,z, vx,vy,vz), 
                              where vr, vt are radial and tangential velocities.
           r_v (bool): if True transform (r, v) --> (x,y,z, vx,vy,vz).
           filename (string): if given a filename, the output will be saved to a text file 
                              the given file name.
           brute (bool): if True find the maximum of the probability function by brute-force (best 
                         for multi-modal functions). Otherwise Nelder-Mead Simplex algorithm 
                         will be used (best for unimodal function).

    Return: (array) sample drawn of the phase variables in (X,V). 
            Return [x,y,z, vx,vy,vz] arrays if either r_vr_vt or r_v is true.
    '''

    
    nX, nV, Xlim, Vlim, = model_class.sampler_input
    nA = 0

    if nX+nV > 6:
        print 'We only support a maximum of 6 variables'
        return
    if (r_vr_vt and r_v)==True:
        print 'You cannot have both r_vr_vt AND r_v set to True'
        return
    if (Xlim[1] <= Xlim[0] or Vlim[1] <= Vlim[0]):
        print 'ERROR: rmax <= rmin or vmax<=vmin, please double check the sample limits'

    ans = rejectsample(model_class.DF, Xlim, Vlim, nX, nV, nA,
                [], [], samplesize, r_vr_vt, r_v, z_vr_vt=False)

    if filename != None:
        if (r_vr_vt or r_v ):
            np.savetxt(filename, np.c_[ans])
        else:
            np.savetxt(filename, np.c_[np.transpose(ans)])

    return ans




###------------------------------------------------------------------------------####
###                                Support Functions                             ####
###------------------------------------------------------------------------------####
def imptsample(fprob, rlim, vlim, Xn, Vn, An, 
                model_param, context, steps, samplesize, rfactor, replace=True, 
                r_vr_vt=False, r_v=False, z_vr_vt=False):

        showtime = True
        t0=time.time()
        if showtime:
                print ' '
                print 'Using importance sampling... '
                print '  building proposal step function... '

        steps = steps+1
        r0 = 1e-9
        rmin, rmax = rlim
        vmin, vmax = vlim
        dx = (rmax - rmin) / (steps-1.0)
        dv = (vmax - vmin) / (steps-1.0)
        da = (2*np.pi)/(steps-1.0)

        #optvar, fmax = getfmax(fprob, Xn, Vn, An, model_param, context, rlim, vlim)
        #optvar = abs(optvar)
        #optvar[0:Xn], optvar[Xn:Vn+Xn], optvar[Vn+Xn:var_num]
        #print 'fmax is: ', optvar, fmax
        
        #build a list of arrays of variables
        gridlist = []
        max_index = []
        for i in range(Xn):
                arr = np.linspace(rmin+r0, rmax, steps)
                #max_index.append( sum(arr < optvar[0:Xn][i])-1 )
                gridlist.append( arr )
        for i in range(Vn):
                arr = np.linspace(vmin, vmax, steps)
                #max_index.append( sum(arr < optvar[Xn:Vn+Xn][i])-1 )
                gridlist.append( arr )
        for i in range(An):
                arr = np.linspace(0, 2*np.pi, steps)
                #max_index.append( sum(arr < optvar[Vn+Xn:var_num][i])-1 )
                gridlist.append( arr )


        #build meshgrid of variables; dimension depends on the number of variables
        s = steps
        var_num = Xn + Vn + An
        if (var_num == 1):
             mgridlist = np.meshgrid( gridlist[0][1:s], indexing='ij')

        if (var_num == 2):
            mgridlist = np.meshgrid( gridlist[0][1:s], gridlist[1][1:s], indexing='ij')

        if (var_num == 3):
            mgridlist = np.meshgrid( gridlist[0][1:s], gridlist[1][1:s], gridlist[2][1:s], indexing='ij')

        if (var_num == 4):
             mgridlist = np.meshgrid( gridlist[0][1:s], gridlist[1][1:s], gridlist[2][1:s], 
                                      gridlist[3][1:s], indexing='ij')

        if (var_num == 5):
             mgridlist = np.meshgrid( gridlist[0][1:s], gridlist[1][1:s], gridlist[2][1:s], 
                                      gridlist[3][1:s], gridlist[4][1:s], indexing='ij')

        if (var_num == 6):
             mgridlist = np.meshgrid( gridlist[0][1:s], gridlist[1][1:s], gridlist[2][1:s], 
                                      gridlist[3][1:s], gridlist[4][1:s], gridlist[5][1:s],indexing='ij')

        #calculate values at grid points
        X = [ Xi-dx*0.5 for Xi in mgridlist[0:Xn]]
        V = [ Vi-dv*0.5 for Vi in mgridlist[Xn:Vn+Xn]]
        A = [ Ai-da*0.5 for Ai in mgridlist[Vn+Xn: var_num]]
        fvalues = fprob(X, V) #, model_param, context)

        #print 'max fvalues: ' , max(fvalues.flatten())
        fvalues = fvalues + max(fvalues.flatten())*.1 #+ np.amin(fvalues[fvalues>0]) #so that it's non-negative
        parr = fvalues.flatten()

        '''
        if (np.all(np.array(max_index)>=0) and np.all(np.array(max_index) < s-1)):
                flatten_max_index = np.ravel_multi_index(max_index, fvalues.shape)
                #print 'GOT THROUGH? ', flatten_max_index, fmax
                parr[flatten_max_index] = fmax if parr[flatten_max_index]<fmax \
                                                else parr[flatten_max_index]
        '''

        if (np.sum(parr)==0):
                print "ERROR: DF might be zero everywhere, or try to increase the step size."
                #print "model param: ", model_param
                return [[1],[2],[3],[4],[5],[6]]

        t1=time.time()
        if showtime:
                print '  complete proposal function, time used: ', t1-t0, 'sec'
                print '  start drawing samples...'
        proptime = t1-t0

        #draw from flattened fvalues, that's equivalent to draw from multidimensional fvalues
        #because dx and dv are constant. 
        rN = int(rfactor * samplesize)
        #print "len of parr negative???? ", parr<0, len(parr)
        norm_parr = parr/np.sum(parr)
        index = np.random.choice(np.arange(len(parr)) , size=rN, p = norm_parr)
        ps = parr[index]  #functional value at the drawn index

        # get the variable values from the drawn index by: 1. flattening meshgrid of a given variable,
        # 2. add the random values at range [0,dx] (or [0,dv]), since the probability is uniform within
        # each step 
        varlist = []
        for i in range(Xn):
                varlist.append(mgridlist[i].flatten()[index] - np.random.rand(rN)*dx )
        for i in range(Vn):
                varlist.append(mgridlist[i+Xn].flatten()[index] - np.random.rand(rN)*dv )
        for i in range(An):
                varlist.append(mgridlist[i+Xn+Vn].flatten()[index] - np.random.rand(rN)*da )


        # calculate the function values at the points drawn from proposal, and calculate 
        # the importance weight ws
        fs = fprob(varlist[0:Xn], varlist[Xn:Vn+Xn]) # , model_param, context)
        ws = fs/ps

        var1 = np.sum( ((ws-1)**2)/len(ws) )

        print 'Weight Variance: ', np.var(ws), var1

        if (sum(ws)==0):
                print 'ERROR all proposal samples has zero probability >> Exit.'
                ne = 99
                #print "model param: ", model_param
                return [[ne],[ne],[ne],[ne],[ne],[ne]]


        #print "less than zero!!!??? ", fs[fs<0]
        auxarr = np.vstack(tuple(varlist))
        samplearr = resample( ws, auxarr, int(samplesize), replace)

        if showtime:
                print '  sampling completed, time used: ', time.time()-t1, 'sec'
                print 'sample time: ', proptime + time.time()-t1,  'sec'
                print '------------------------------------'

        if r_vr_vt:
                return r_vr_vt_complete(samplearr)
        if r_v:
                return r_v_complete(samplearr)
        if z_vr_vt:
                return z_vr_vt_complete(samplearr)

        return np.transpose( np.array(samplearr) )


def resample(warr, auxarr, samplesize, replace):
        auxarr2 = np.transpose(auxarr)
        warr = warr/sum(warr)
        index = np.random.choice(np.arange(len(warr)), size=samplesize, replace=replace, p=warr)
        result2 = auxarr2[index]
        return result2


def rejectsample(fprob, rlim, vlim, Xn, Vn, An, 
                model_param, context, samplesize, r_vr_vt=False, r_v=False, z_vr_vt=False, brute=True):

        showtime = True
        t0=time.time()
        if showtime:
                print ' '
                print 'Using rejection sampling... '
        

        var_num = Xn + Vn + An
        samplesize = int(samplesize)        
        r0   = 10**-9
        rmin, rmax = rlim
        vmin, vmax = vlim

        optvar, fmax = getfmax(fprob, Xn, Vn, An, model_param, context, rlim, vlim, brute)
        #print 'fmax is: ', optvar, fmax
        #fmax=1e-5
        
        # rejection sampling loop. Pre-evaluate a number of function values to 
        # estimate acceptance rate, then calculate the function 
        # auxN = N_sample_still_needed/acceptance_rate number of times, 
        # keep doing that until number of samples has reached @samplesize
        acceptX = []
        acceptV = []
        acceptA = []
        samplelist = []
        num  = samplesize
        auxN = samplesize
        eff = 1.0
        output = np.array([])
        acceptN = 0
        computeN = 0
        j = 0
        uX = []
        while ( True ):
                auxN = int(auxN/(eff))
                #for i in range(Xn):
                #        uX.append( np.random.rand(auxN)*rlim )
                if (auxN > 1e7):
                        auxN = int(1e7)
                uX = ( np.random.rand(Xn,auxN)*(rmax-rmin) + rmin )  
                uV = ( np.random.rand(Vn,auxN)*(vmax-vmin) + vmin )
                uA = ( np.random.rand(An,auxN)*np.pi*2 )
                u  = np.random.rand(auxN)

                uf = fprob(list(uX), list(uV))
                accept_index = ( (uf/fmax)>u )
                acceptN += np.sum(accept_index)
                computeN += auxN
                eff = acceptN/(computeN*1.0)
                eff = eff if eff>0 else .01

                acceptX.append( uX[:,accept_index] )
                acceptV.append( uV[:,accept_index] )
                acceptA.append( uA[:,accept_index] )

                print '  ', j, ' total_points_proposed: ', computeN, ' total_points_accepted: ', acceptN
                j=j+1

                if (acceptN >= samplesize):
                        break
                auxN = samplesize - acceptN

        Xarr = np.hstack(tuple(acceptX))[:,0:samplesize]
        Varr = np.hstack(tuple(acceptV))[:,0:samplesize]
        Aarr = np.hstack(tuple(acceptA))[:,0:samplesize]

        samplearr = np.vstack((Xarr,Varr,Aarr))
        samplearr = np.swapaxes(samplearr,0,1)

        if showtime:
                print "  final acceptance rate: ", eff
                print '  sample time: ', time.time()-t0, 'sec'
                print '------------------------------------'

        if r_vr_vt:
                return r_vr_vt_complete(samplearr)
        if r_v:
                return r_v_complete(samplearr)
        if z_vr_vt:
                return z_vr_vt_complete(samplearr)

        return np.transpose(samplearr)

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

#give random direction to both velocity and radial position, convert to cartesian coordinate
def r_v_complete(samplelist):
        #into x, y, z, vx, vy, vz
        samplelist = np.asarray(samplelist)
        x = samplelist[:,0]
        rsize = len(x)
        ru = np.random.random_sample( rsize )
        rtheta = np.arccos(1-2*ru)
        rphi   = np.random.random_sample( rsize ) * 2.0 * np.pi

        xx = x * np.sin(rtheta) * np.cos(rphi)
        yy = x * np.sin(rtheta) * np.sin(rphi)
        zz = x * np.cos(rtheta)

        v = samplelist[:,1]
        vu     = np.random.random_sample( rsize )
        vtheta = np.arccos(1-2*vu)
        vphi   = np.random.random_sample( rsize ) * 2.0 * np.pi
        vx     = v * np.sin(vtheta) * np.cos(vphi)
        vy     = v * np.sin(vtheta) * np.sin(vphi)
        vz     = v * np.cos(vtheta)

        return xx, yy, zz, vx,vy,vz


def z_vr_vt_complete(samplelist, context):
        #Important: to use z_vr_vt, the fist element of context needs to be the projected radius R.
        R = context[0]
        if len(samplelist)>=1:
                samplearr = np.asarray(samplelist)
                zarr  = samplearr[:,0]
                vrarr = samplearr[:,1]
                vtarr = samplearr[:,2]

        else:   #in case it's empty..
                print "ERROR: Empty sampleset"

        zsize = len(zarr)
        phi = np.random.random_sample( zsize ) * 2.0 * np.pi
        x   = R * np.cos(phi)
        y   = R * np.sin(phi)
        z   = zarr

        r  = (z*z + x*x + y*y)**.5
        theta = np.arccos(z/r) 

        v  = (vrarr*vrarr + vtarr*vtarr )**0.5
        vsign = np.sign( np.random.random_sample( zsize ) - 0.5 )  #probably not necessary?
        vphi  = np.random.random_sample( zsize ) * 2.0 * np.pi

        vz2 = vsign * vrarr
        vx2 = vtarr * np.cos(vphi)
        vy2 = vtarr * np.sin(vphi)

        #passive rotation, using rotation matrix 
        #to rotate the zhat of calculate velocity into the zhat of the spatial coordinate
        vx = np.cos(theta)*np.cos(phi)*vx2 - np.sin(phi)*vy2 + np.sin(theta)*np.cos(phi)*vz2
        vy = np.cos(theta)*np.sin(phi)*vx2 + np.cos(phi)*vy2 + np.sin(theta)*np.sin(phi)*vz2
        vz = -np.sin(theta)*vx2 + np.cos(theta)*vz2
        return x, y, z, vx, vy, vz

        

def getfmax(fprob, Xn, Vn, An, model_param, context, rlim, vlim, brute=True):

        rmax = rlim[1]
        vmax = vlim[1]
        var_num = Xn + Vn + An
        guessX = np.array( [rmax*.1]*Xn ) #+ 1e-3
        guessV = np.array( [vmax*.1]*Vn ) #+ 1e-3
        guess = tuple( np.hstack((guessX, guessV)) )
        
        range = tuple( [rlim]*Xn + [vlim]*Vn ) #note rlim and vlim is [min, max]


        def aux_fprob(*args):
            result = fprob(args[0][0:Xn], args[0][Xn:Vn+Xn])
            return -1*result

        if brute:
            optvar = scipy.optimize.brute(aux_fprob,
                       range, args=(), Ns=20, full_output=True, finish=scipy.optimize.fmin)[0]
        else:
            optvar = scipy.optimize.fmin(aux_fprob,
                       guess, xtol=1e-7, ftol=1e-7, maxiter=int(1e6), maxfun=int(1e7), disp=False)


        fmax = 1.0 * fprob(optvar[0:Xn], optvar[Xn:Vn+Xn]) 
        print 'optvar, max: ', optvar, fmax

        return optvar, fmax*1.05 +00 #TODO



