



class Model(object):
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


        # Required user input regarding this specific model
        self.nX = 
        self.nV = 
        self.Xlim = [ , ] 
        self.Vlim = [ , ] 

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
        
        #r = X #unpack X
        #v  = V #unpack V

        # calculate probability


        return #probability





