# StarSampler

StarSampler is a Python framework that generates random samples from any user-defined distribution function(DF) that specifies the probability density of stellar coordinates within six-dimensional phase space.
Users need to define a density '''DF''' and an '''input\_function'''. The reasons for the input\_function are, 1: pre-compute elements needed for density function calculation; 2: provide information needed for the StarSampler, which include number of spatial and velocity coordinates and spatial and velocity range. The functions needs to have the following signitures and outputs.

'''python
def user\_DF(X, V, model\_param, context):
  #X: list of input position coordinates (i.e. [radius] or [x,y,z]).
  #V: list of input velocity coordinates (i.e. [tangential_speed, radial_speed] or [vx,vy,vz]).
  #model\_param: list of parameters for your model.
  #context: a list of additional items user wish to pass to this density function.

return #DF\_probability
'''
AND 
'''python
def sampler\_input(model\_param):
  #same model\_param as in the user\_DF fucntion, a list
  #nX: number of spatial coordinates, an int
  #nV: number of velocity coordinates, an int
  #context: a list of items that will be passed directly to the user\_DF 
  #rlim: sample range of spatial coordinates;  i.e. [lower-limit, upper-limit]
  #vlim: sample range of velocity coordinates; i.e. [lower-limit, upper-limit]
  
return nX, nV, context, rlim, vlim
'''

We provide three stellar DF examples: King model, Osipkov-Merritt, and SFW. See the paper for the descriptions of each models.

For example, to sample from the SFW model, after defining DF and the input function, as in the sfw.py.
'''python
import star_sampler as ssp
import sfw

#specify model parameters
param = [2.0, -5.3, 2.5, 0.16, 1.5, 0,-9.0 , 1, 6.9, 0.086, .078, 0.694444, 1., 3., 1.]
a,d,e, Ec, rlim, b0,b1, alp, q, Jb, rho, rs, alpha, beta, gamma = param
model\_param = [a, d,  e, Ec, rlim, b0,b1,alp, q, Jb, rho, rs, alpha, beta, gamma]

Nstars = 1000 #the number of samples to draw

#1. construct the sampler with model\param, provide the DF function and input function
ssam = ssp.Sampler(myDF = sfw.sfw_fprob, sampler_input = sfw.sampler\_input, model\_param=model\_param)

#2. sample the model using rejection sampling, specify filename[string] will save the output to the file.
#@params: r_vr_vt=False, r_v=False; setting one of them to True will activate
#corresponding transformation to [x,y,z,vx,vy,vz] coordinates.
sfw\_rej\_samples = ssam.sample(sample\_method='rejection', N=Nstars, filename=None, r_vr_vt=False, r_v=False)

#OR
#3. sample the model using importance sampling, requires additional @param steps and @param rfactor.
#@param steps is number of steps for the proposal function, and rfactor is a multiplication factor that sets the number of proposal points to draw based on the desired samples.

sfw\_impt\_samples = ssam.sample(sample\_method='impt', N = Nstars, steps = 20, rfactor = 3,
                filename=None, r_vr_vt=True, r_v=False)
                
'''

To sample from King model,
'''python
import star_sampler as ssp
import king as k

sigma = 10
ratio0 = 9
psi0 = (sigma**2) * ratio0
model_param = [psi0, ratio0]
Nstars = 1e4

#1. construct the sampler
ssam = ssp.Sampler(myDF = k.king_fprob, sampler_input = k.sampler_input,
                        model_param=model_param)

#2. sample the model using rejection sampling
x1,y1,z1,vx1,vy1,vz1 = ssam.sample(sample_method='rejection', N=Nstars, filename=None,
                r_vr_vt=False, r_v=True)

#3  sample the model using importance sampling
x2,y2,z2,vx2,vy2,vz2 = ssam.sample(sample_method='impt', N = Nstars, steps = 20, rfactor = 3,
                filename=None, r_vr_vt=False, r_v=True)
'''

To sample from Osipkov_Merritt model,
'''python
import star_sampler as ssp
import osipkov_merritt as om

model_param = [1.0, 1., 2.,5., .1, .4 , 5., 1, 3.5, 1]
Nstars= 1000

#1. construct the sampler
ssam = ssp.Sampler(myDF=om.OM_fprob, sampler_input = om.sampler_input,
                        model_param=model_param)

#2. using rejection sampling
rej_output = ssam.sample(sample_method='rejection', N=Nstars, filename='om_rej.txt', r_vr_vt=True, r_v=False)
x1,y1,z1,vx1,vy1,vz1 = rej_output

#3, Or use importance sampling.
impt_output = ssam.sample(sample_method='impt', N=Nstars, steps=20, rfactor=30, filename='om_impt.txt', r_vr_vt=True, r_v=False)
x2,y2,z2,vx2,vy2,vz2 = impt_output
'''





