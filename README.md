# StarSampler

StarSampler is a Python module that generates random samples from any user-defined distribution function(DF) that specifies the probability density of stellar coordinates within six-dimensional phase space. Two sampling methods are available, the *rejection sampling* and the *importance sampling*. See the attached paper for more details. 


### Specify Density Function and Input parameters
User needs to define a Model `class`, which includes the `__init__(self, **model_param)` and a `DF(self, X,V)` functions. Within the `__init__()` function user need to provide information needed for the StarSampler, which include number of spatial and velocity coordinates `(nX, nV)`, and spatial and velocity range `(Xlim, Vlim)`. User can also pre-compute elements needed for density function calculation.

The `__init__()` function should contains following items,

```python
def __init__(self, **model_param):
  #model_param: a dictionary or keyword arguments that specify the stellar distribution.
  
  #nX: an int, the number of spatial coordinates
  #nV: an int, the number of velocity coordinates
  #Xlim: sample range of spatial coordinates;  i.e. [lower-limit, upper-limit]
  #Vlim: sample range of velocity coordinates; i.e. [lower-limit, upper-limit]
   
  # Required: 
  self.sampler_input = [self.nX, self.nV, self.Xlim, self.Vlim]
```

AND 

The `DF()` functions needs to have the following signitures and outputs. Model parameters and other elements needed for the probability calculation can be directly accessed from the Model class.
```python
def DF(self, X, V):
  #X: list of input position coordinates (i.e. [radius] or [x,y,z]).
  #V: list of input velocity coordinates (i.e. [tangential_speed, radial_speed] or [vx,vy,vz]).
  
  return #DF_probability
```

See examples of the three stellar DFs: King model, Osipkov-Merritt, and SFW in the StarSampler directory. A brief discription of each model is in the paper.




### Sample from a Distribution Function

As an example, after defining the *SFW* model class and its`DF`, as in the *sfw.py*, to sample from the model:

```python
import star_sampler as ssp
import sfw

#specify sfw model parameters
model_param = {'a':2, 'd':-5.3, 'e':2.5, 'Ec':.16, 'rlim':1.5, 'b0':0, 'b1':-9, 'alp':1,
               'q':6.9, 'Jb':.086, 'rho':7.8e7, 'rs':.694, 'alpha':1, 'beta':3, 'gamma':1}

Nstars = 1000 #the number of samples to draw

#1. construct the SFW model
sfw1 = sfw.SFW(**model_param)

#2. sample the model using rejection sampling .
# specify filename (a string) will save the output to the file.
# @params: r_vr_vt=False, r_v=False; if the DF has (r, vr, vt) or (r, v) as the coordinates, 
# user can set one of them to True and that will activate corresponding transformation 
# to [x,y,z,vx,vy,vz] coordinates.

sfw_rej_samples = ssp.rejection_sample(sfw1, samplesize = Nstars, 
                                       r_vr_vt=True, r_v=False, filename=None)

#OR
#3. sample the model using importance sampling, requires two additional parameters.
# @param steps: number of steps for the proposal function, 
# @param rfactor: the multiplication factor of the sample size that sets the number of
#                 proposal points to draw.

sfw_impt_samples = ssp.impt_sample(sfw1,  steps=20, resample_factor=5,
                samplesize = Nstars, replace=True, r_vr_vt=True, r_v=False, filename=None)               
```



We follow the same procedure to sample from *King* model, defined in *king.py*.

```python
import star_sampler as ssp
import king as k

# set model parameters
sigma = 10 #unit [km/s]
ratio0 = 9  
rho0 = 1e8 #unit [M_sun / kpc^3]
model_param = {'sigma':sigma, 'ratio0':ratio0, 'rho0':rho0}

Nstars = 1e4

#1. construct the King model
king1 = k.King(**model_param)

#2. sample the model using rejection sampling
x1,y1,z1,vx1,vy1,vz1 = ssp.rejection_sample(king1, samplesize = Nstars,
                r_vr_vt=False, r_v=True, filename=None) 

#3  sample the model using importance sampling
x2,y2,z2,vx2,vy2,vz2 = ssp.impt_sample(king1, steps=20, resample_factor=5,
                samplesize = Nstars, replace=True, r_vr_vt=False, r_v=True, filename=None)
```



To sample from *Osipkov\_Merritt\_Zhao* model, defined in *osipkov_merritt.py*

```python
import star_sampler as ssp
import osipkov_merritt as om

model_param = {'ra': 0.1, 'rs_s':0.1, 'al_s':2, 'be_s':5, 'ga_s':.1,
               'rho':.064*1e9, 'rs':1.0, 'alpha':1., 'beta':3., 'gamma':1.}
Nstars= 1000

#1. construct the OM model
om1 = om.OM(**model_param)

#2. using rejection sampling
x1,y1,z1,vx1,vy1,vz1 = ssp.rejection_sample(om1, samplesize = Nstars,
                r_vr_vt=True, filename=None)
                
#3, Or use importance sampling.
x2,y2,z2,vx2,vy2,vz2 = ssp.impt_sample(om1, steps=20, resample_factor=5,
                samplesize = Nstars, replace=True, r_vr_vt=True, filename=None)
```

For some combinations of *Osipkov\_Merritt* model parameters the sampling can become prohibitively inefficient. There is a additional standalone conditional sampling routine (within the OM model class) with better sampling efficiency. To draw samples using this routine,

```python
import osipkov_merritt as om

model_param = {'ra': 0.1, 'rs_s':0.1, 'al_s':2, 'be_s':5, 'ga_s':.1,
               'rho':.064*1e9, 'rs':1.0, 'alpha':1., 'beta':3., 'gamma':1.}
Nstars= 1000

# model construction
om1 = om.OM(**model_param)

# Use OM conditional sampler
#@param Phi_table_steps: number of steps that calculates the potential for 
#                        interpolation. (1e5 is good)
#@param GQ_table_steps: number of steps that calculates the G(Q) function for 
#                       interpolation and take derivatives. (1000 is sufficent)
#@param proposal_steps: number of steps to calculate to construct the proposal 
#                       density of both r and Q distribution. (1000 is sufficient)

OM_conditional_output = om.conditional_sample(samplesize=Nstars, \
    Phi_table_steps=1e5, GQ_table_steps=1000, proposal_steps = 1000, r_vr_vt=True)
    
```


