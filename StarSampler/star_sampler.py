import numpy as np
import random as rand
from scipy import integrate
import scipy.optimize
import scipy.special as ss
import time



class Sampler(object):
    def __init__(self, myDF=None, sampler_input=None, model_param=None):
	#R=None
        self.nX, self.nV, self.context, self.rlim, self.vlim = sampler_input(model_param)


	if self.nX+self.nV > 6:
		print 'We only support maximum of 6 variables'
		return

	self.model_param = model_param
	self.myDF = myDF
	if myDF==None or sampler_input==None:
	    print 'Please specify your DF and/or sampler_input function(s)'
            return
	
	if (self.rlim[1] <= self.rlim[0] or self.vlim[1] <= self.vlim[0]):
            print 'ERROR: rmax <= rmin or vmax<=vmin, please double check the sample limits'


    def sample(self, sample_method='rejection', N=1000, steps = 20, rfactor = 3, 
		r_vr_vt=False, r_v=False, filename=None):

	if sample_method == 'rejection' :
            ans = rejectsample(self.myDF, self.rlim, self.vlim, self.nX, self.nV, 0,
                            self.model_param, self.context, N, r_vr_vt, r_v)

	if sample_method == 'impt' :
            ans = imptsample(self.myDF, self.rlim, self.vlim, self.nX, self.nV, 0,
                            self.model_param, self.context, steps, N, rfactor, r_vr_vt, r_v)

        if filename != None: 
	    if (r_vr_vt or r_v ):
		np.savetxt(filename, np.c_[ans], fmt='%1.6f')
	    else:
		
		np.savetxt(filename, np.c_[np.transpose(ans)], fmt='%1.6f')

        return ans


###------------------------------------------------------------------------------####
###				Support Functions				 ####
###------------------------------------------------------------------------------####
def imptsample(fprob, rlim, vlim, Xn, Vn, An, 
		model_param, context, steps, samplesize, rfactor, 
		r_vr_vt=False, r_v=False, z_vr_vt=False):

	showtime = True
	t0=time.time()
	if showtime:
		print ' '
		print 'Using importance sampling... '
		print '  building proposal step function... '

	r0 = 1e-9
	rmin, rmax = rlim
	vmin, vmax = vlim
        dx = (rmax - rmin) / (steps-1.0)
        dv = (vmax - vmin) / (steps-1.0)
	da = (2*np.pi)/(steps-1.0)

	optvar, fmax = getfmax(fprob, Xn, Vn, An, model_param, context, rmax, vmax)
	optvar = abs(optvar)
	#optvar[0:Xn], optvar[Xn:Vn+Xn], optvar[Vn+Xn:var_num]
	#print 'fmax is: ', optvar, fmax
	
	#build a list of arrays of variables
	gridlist = []
	max_index = []
	for i in range(Xn):
		arr = np.linspace(rmin+r0, rmax, steps)
		max_index.append( sum(arr < optvar[0:Xn][i])-1 )
		gridlist.append( arr )
	for i in range(Vn):
		arr = np.linspace(vmin, vmax, steps)
		max_index.append( sum(arr < optvar[Xn:Vn+Xn][i])-1 )
		gridlist.append( arr )
	for i in range(An):
		arr = np.linspace(0, 2*np.pi, steps)
		max_index.append( sum(arr < optvar[Vn+Xn:var_num][i])-1 )
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
	fvalues = fprob(X, V, model_param, context)
	fvalues = fvalues + fmax*.05 #+ np.amin(fvalues[fvalues>0]) #so that it's non-negative
        parr = fvalues.flatten()

	if (np.all(np.array(max_index)>=0) and np.all(np.array(max_index) < s-1)):
		flatten_max_index = np.ravel_multi_index(max_index, fvalues.shape)
		#print 'GOT THROUGH? ', flatten_max_index, fmax
		parr[flatten_max_index] = fmax

	if (np.sum(parr)==0):
		print "ERROR: density function is zero everywhere."
                print "model param: ", model_param
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
	fs = fprob(varlist[0:Xn], varlist[Xn:Vn+Xn], model_param, context)
        ws = fs/ps

        if (sum(ws)==0):
                print 'ERROR all proposal samples has zero probability'
		ne = 99
                print "model param: ", model_param
                return [[ne],[ne],[ne],[ne],[ne],[ne]]


	#print "less than zero!!!??? ", fs[fs<0]
        auxarr = np.vstack(tuple(varlist))
        samplearr = resample( ws, auxarr, int(samplesize) )

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


def resample(warr, auxarr, samplesize):
	auxarr2 = np.transpose(auxarr)
        warr = warr/sum(warr)
        index = np.random.choice(np.arange(len(warr)), size=samplesize, p=warr)
	result2 = auxarr2[index]
        return result2


def rejectsample(fprob, rlim, vlim, Xn, Vn, An, 
		model_param, context, samplesize, r_vr_vt=False, r_v=False, z_vr_vt=False):

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

	optvar, fmax = getfmax(fprob, Xn, Vn, An, model_param, context, rmax, vmax)
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
		#	uX.append( np.random.rand(auxN)*rlim )
		if (auxN > 1e7):
			auxN = int(1e7)
		uX = ( np.random.rand(Xn,auxN)*(rmax-rmin) + rmin )  
		uV = ( np.random.rand(Vn,auxN)*(vmax-vmin) + vmin )
		uA = ( np.random.rand(An,auxN)*np.pi*2 )
                u  = np.random.rand(auxN)

                uf = fprob(list(uX), list(uV), model_param, context)
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

	

def getfmax(fprob, Xn, Vn, An, model_param, context, rmax, vmax):

        var_num = Xn + Vn + An
	guessX = np.array( [rmax*.1]*Xn )
	guessV = np.array( [vmax*.1]*Vn )
	guess = tuple( np.hstack((guessX, guessV)) )
	

        if var_num == 6 :
            def aux_fprob(x1,x2,x3,x4,x5,x6):
                vars = [x1,x2,x3,x4,x5,x6]
                result = fprob(vars[0:Xn], vars[Xn:Vn+Xn],
                                model_param, context)
                return result
            optvar = scipy.optimize.fmin(lambda (x1,x2,x3,x4,x5,x6): -aux_fprob(x1,x2,x3,x4,x5,x6),
                                                (0.1,0.1,0.1,0.1,0.1,0.1), disp=False)

        if var_num == 5 :
            def aux_fprob(x1,x2,x3,x4,x5):
                vars = [x1,x2,x3,x4,x5]
                result = fprob(vars[0:Xn], vars[Xn:Vn+Xn], 
                                model_param, context)
                return result
            optvar = scipy.optimize.fmin(lambda (x1,x2,x3,x4,x5): -aux_fprob(x1,x2,x3,x4,x5),
                                                (0.1,0.1,0.1,0.1,0.1), disp=False)

        if var_num == 4 :
            def aux_fprob(x1,x2,x3,x4):
                vars = [x1,x2,x3,x4]
                result = fprob(vars[0:Xn], vars[Xn:Vn+Xn], 
                                model_param, context)
                return result
            optvar = scipy.optimize.fmin(lambda (x1,x2,x3,x4): -aux_fprob(x1,x2,x3,x4),
                                                (0.1,0.1,0.1,0.1), disp=False)

        if var_num == 3 :
            def aux_fprob(x1,x2,x3):
                vars = [x1,x2,x3]
                result = fprob(vars[0:Xn], vars[Xn:Vn+Xn], 
                                model_param, context)
                return result
            optvar = scipy.optimize.fmin(lambda (x1,x2,x3): -aux_fprob(x1,x2,x3),
                                                (0.1,0.1,0.1), disp=False)

        if var_num == 2 :
            def aux_fprob(x1,x2):
                vars = [x1,x2]
                result = fprob(vars[0:Xn], vars[Xn:Vn+Xn], 
                                model_param, context)
                return result
            optvar = scipy.optimize.fmin(lambda (x1,x2): -aux_fprob(x1,x2),
                                                guess, disp=False)

	if var_num == 1 :
            def aux_fprob(x1):
                vars = [x1]
                result = fprob(vars[0:Xn], vars[Xn:Vn+Xn], 
                                model_param, context)
                return result
            optvar = scipy.optimize.fmin(lambda (x1): -aux_fprob(x1), (0.1), disp=False)



        fmax = 1.0 * fprob(optvar[0:Xn], optvar[Xn:Vn+Xn], 
                                model_param, context)

        return optvar, fmax*1.05
