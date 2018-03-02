# StarSampler

StarSampler is a Python framework that generates random samples from any user-defined distribution function(DF) that specifies the probability density of stellar coordinates within six-dimensional phase space. Two sampling methods are available, the *rejection sampling* and the *importance sampling*. See the [paper](https://arxiv.org/) for more details. 


### Specify Density Function and Input parameters
User needs to define a density `DF` and a `sampler_input` functions. The reasons for the `sampler_input` function are:

1: Provide information needed for the StarSampler, which include number of spatial and velocity coordinates, and spatial and velocity range. 

2: Pre-compute elements needed for density function calculation.


The functions needs to have the following signitures and outputs.
```python
def DF(X, V, model_param, context):
  #X: list of input position coordinates (i.e. [radius] or [x,y,z]).
  #V: list of input velocity coordinates (i.e. [tangential_speed, radial_speed] or [vx,vy,vz]).
  #model_param: list of parameters for your model.
  #context: a list of additional items user wish to pass to this density function.

return #DF_probability
```

AND 

```python
def sampler_input(model_param):
  #model_param: a list, as in the DF fucntion
  #nX: an int, the number of spatial coordinates
  #nV: an int, the number of velocity coordinates
  #context: a list of items that will be passed directly to the DF function
  #rlim: sample range of spatial coordinates;  i.e. [lower-limit, upper-limit]
  #vlim: sample range of velocity coordinates; i.e. [lower-limit, upper-limit]
  
return nX, nV, context, rlim, vlim
```

See examples of the three stellar DFs: King model, Osipkov-Merritt, and SFW in the StarSampler directory. A brief discription of each model is in the paper.



### Sample from a Distribution Function

As an example, after defining the *SFW* `DF` and the corresponding `sampler_input` functions, as in the *sfw.py*, to sample from the model:

```python
import star_sampler as ssp
import sfw

#specify sfw model parameters
param = [2.0, -5.3, 2.5, 0.16, 1.5, 0,-9.0 , 1, 6.9, 0.086, .078, 0.694444, 1., 3., 1.]
a,d,e, Ec, rlim, b0,b1, alp, q, Jb, rho, rs, alpha, beta, gamma = param
model_param = [a, d,  e, Ec, rlim, b0,b1,alp, q, Jb, rho, rs, alpha, beta, gamma]

Nstars = 1000 #the number of samples to draw

#1. construct the sampler with model_param, input the DF function and input function
ssam = ssp.Sampler(myDF = sfw.sfw_fprob, sampler_input = sfw.sampler_input, model_param=model_param)

#2. sample the model using rejection sampling .
# specify filename (a string) will save the output to the file.
# @params: r_vr_vt=False, r_v=False; if the DF has (r, vr, vt) or (r, v) as the coordinates, 
# user can set one of them to True and that will activate corresponding transformation 
# to [x,y,z,vx,vy,vz] coordinates.

sfw_rej_samples = ssam.sample(sample_method='rejection', N=Nstars, 
                                filename=None, r_vr_vt=False, r_v=False)

#OR
#3. sample the model using importance sampling, requires additional @param steps and @param rfactor.
# @param steps: number of steps for the proposal function, 
# @param rfactor: the multiplication factor of the sample size that sets the number of proposal 
#                 points to draw.

sfw_impt_samples = ssam.sample(sample_method='impt', N = Nstars, steps = 20, rfactor = 3,
                                filename=None, r_vr_vt=True, r_v=False)
                
```


We follow the same procedure to sample from *King* model, defined in *king.py*.

```python
import star_sampler as ssp
import king as k

# set model parameters
sigma = 10
ratio0 = 9
psi0 = (sigma**2) * ratio0
model_param = [psi0, ratio0]
Nstars = 1e4

#1. construct the sampler
ssam = ssp.Sampler(myDF = k.king_fprob, sampler_input = k.sampler_input,
                    model_param=model_param)

#2. sample the model using rejection sampling
x1,y1,z1,vx1,vy1,vz1 = ssam.sample(sample_method='rejection', N=Nstars, 
                    filename=None, r_vr_vt=False, r_v=True)

#3  sample the model using importance sampling
x2,y2,z2,vx2,vy2,vz2 = ssam.sample(sample_method='impt', N = Nstars, steps = 20, rfactor = 3,
                    filename=None, r_vr_vt=False, r_v=True)
```


To sample from *Osipkov\_Merritt* model, defined in *osipkov_merritt.py*

```python
import star_sampler as ssp
import osipkov_merritt as om

model_param = [1.0, 1., 2.,5., .1, .4 , 5., 1, 3.5, 1]
Nstars= 1000

#1. construct the sampler
ssam = ssp.Sampler(myDF=om.OM_fprob, sampler_input = om.sampler_input,
                        model_param=model_param)

#2. using rejection sampling
rej_output = ssam.sample(sample_method='rejection', N=Nstars, 
                          filename='om_rej.txt', r_vr_vt=True, r_v=False)
x1,y1,z1,vx1,vy1,vz1 = rej_output

#3, Or use importance sampling.
impt_output = ssam.sample(sample_method='impt', N=Nstars, steps=20, rfactor=30, 
                          filename='om_impt.txt', r_vr_vt=True, r_v=False)
x2,y2,z2,vx2,vy2,vz2 = impt_output
```

For some combinations of *Osipkov\_Merritt* model parameters the sampling can become prohibitively inefficient. There is a additional standalone conditional sampling routine with better sampling efficiency. To draw samples using this routine,

```python
import osipkov_merritt as om
model_param = [1.0, 1., 2.,5., .1, .4 , 5., 1, 3.5, 1]
Nstars= 1000

#@param Phi_table_steps is number of steps that calculates the potential for interpolation.
#@param GQ_table_steps is number of steps that calculates the G(Q) function for interpolation, 
#   and take derivative.
#@param proposal_steps is number of steps to calculate to construct the proposal density of 
#   both r and Q distribution.
OM_conditional_output = om.OM_sample(model_param, Nstars, \
    Phi_table_steps=1e5, GQ_table_steps=1000, proposal_steps = 1000, r_vr_vt=True)

```


