# Sub Optimizer
# version 1.6 (2022/01/12)

# standard library
import sys
import math             as mt
# additional library
import numpy            as np
import pandas           as pd
from utils              import log
''' For Function Annotations '''
from config     import Configuration
from function   import Function

class Population:
    '''population class for optimizer
    '''
    def __init__(self) -> None:
        '''initialization of all variables
        '''
        # population
        self.x      = None
        self.f      = None
        # current log
        self.x_new     = None
        self.f_new     = None
        # best log
        self.x_best     = None
        self.f_best     = None

    @property
    def getBest(self) -> tuple:
        '''Best data getter method (external reference)

        Returns:
            tuple: best x and f
        '''
        return self.x_best, self.f_best

    @property
    def getCurrent(self) -> tuple:
        '''Current data getter method (external reference)
        Returns:
            tuple: current x and f
        '''
        return self.x_new, self.f_new

    @property
    def getPopulation(self) -> tuple:
        '''Population getter method (external reference)
        Returns:
            tuple: x and f
        '''
        return self.x, self.f


class OptimizerCore:
    '''Core optimizer method
    '''
    def __init__(self, cnf:Configuration, fnc:Function) -> None:
        self.cnf                = cnf
        self.fnc                = fnc
        # update index
        self.indices            = { 'div': 0, 'pop': 0 }
        # group
        self.group              = []
        self.dim                = []
        # grouping name
        self.static_grouping    = ['SG']
        self.random_grouping    = ['RG']
        self.dynamic_grouping   = ['RDG3']


    '''
        Optimization (Minimize & Maximize)
    '''
    def setOptType(self, fnc_type:str) -> None:
        '''Guarantee optimization (minimization/maximization) judgment
        '''
        max_prob = []       # maximization problem lists
        if fnc_type in max_prob:
            self.cnf.opt_type = 'max'
            self.init_fitness = -np.inf
        else:
            self.cnf.opt_type = 'min'
            self.init_fitness = np.inf


    def superior(self, value_1:float, value_2:float) -> bool:
        '''whether value_1 is superior to value_2
        (if value_1 is equal to value_2, function returns False)
        EX) minimize
            value_1 <  value_2 : True
            value_1 >= value_2 : False

        Args:
            value_1 (float): fitness value (like f_new)
            value_2 (float): fitness value (like f_best)

        Returns:
            bool: whether value_1 is superior to value_2
        '''
        if self.cnf.opt_type == 'min':
            return True if value_1 < value_2 else False
        elif self.cnf.opt_type == 'max':
            return True if value_1 > value_2 else False
        else:
            log(self, f'Error: Invalid opt_type "{self.cnf.opt_type}".', output=sys.stderr)
            sys.exit(1)


    def getFitness(self, x:list|tuple|np.ndarray, logging:bool=False) -> tuple|float :
        '''get single/multi fitness value

        Args:
            x (list or tuple or np.ndarray): single/multi solution
                single solution:
                    x : ndarray(dim)
                multi solutions:
                    xs : [x,x,...,x] or (x,x,...,x) or ndarray(x,x,..,x)
                    * x type is ndarray.
            logging (bool): logging fitness value

        Returns:
            float : fitness value of single solution
            tuple : fitness values of multi solution
        '''
        # x is composed of multi decision value
        if logging and not 'f_log' in vars(self).keys():
            self.f_log = []
        if (isinstance(x, np.ndarray) and x.ndim ==2 ) or isinstance(x,(list,tuple)):
            if self.fnc.total_evals >= self.cnf.max_evals:
                return 'Exceed FEs'
            if not logging:
                return tuple(map(self.fnc.doEvaluate,x))
            else:
                self.f_log.extend(f:=tuple(map(self.fnc.doEvaluate,x)))
                return f
        # x is composed of single decision value
        elif isinstance(x, np.ndarray) and x.ndim ==1:
            if not logging:
                return self.fnc.doEvaluate(x)
            else:
                self.f_log.append(f:=self.fnc.doEvaluate(x))
                return f
        else:
            log(self, f'Error: Invalid type argument in getFitness(). {type(x)}', output=sys.stderr)
            return 'Invalid Input'


    def initializePopulation(self, pop:Population) -> Population:
        '''Initialize all population (Normal)

        Args:
            pop (Population): population you want to initialize

        Returns:
            Population: population set by initial values
        '''
        # set variables
        pop_size, dim, axis_range = self.cnf.max_pop, self.fnc.prob_dim, self.fnc.axis_range.copy()
        lb, ub = axis_range[:,0], axis_range[:,1]
        # initialize population
        assert dim == axis_range.shape[0], 'Error: Dimension does not match.'
        assert len(self.dim) == self.max_div, f'Error: Internal variable len(subdim) "{len(self.dim)}" != max_div "{self.max_div}".'
        match self.cnf.init_method:
            case 'random':
                pop.x = [self.cnf.rd.uniform(lb[subgroup], ub[subgroup], (pop_size, subdim)) for subgroup,subdim in zip(self.group,self.dim)]
            case 'lhs':
                from pyDOE import lhs
                pop.x = [
                    lb[subgroup] + (ub[subgroup] - lb[subgroup]) * lhs(subdim,pop_size,'c')
                    for subgroup,subdim in zip(self.group,self.dim)
                    ]
            case _:
                log(self,f'Error: Invalid init_method "{self.cnf.init_method}"')
        pop.f = np.full((pop_size, self.max_div), self.init_fitness)
        pop.x_best = np.full(dim,np.nan)
        pop.f_best = self.init_fitness
        return pop

    def updateBest(self, pop:Population, x_new:np.ndarray, f_new:float) -> Population:
        '''Update best position and fitness value
        '''
        if self.superior(f_new, pop.f_best):
            pop.x_best = x_new.copy()
            pop.f_best = f_new
        return pop


    '''
        Cooperative Co-evolution
    '''
    @property
    def getIndices(self) -> tuple[int,int]:
        '''Get indices of div and pop (external reference)

        Returns:
            tuple: div, pop
        '''
        return self.indices['div'], self.indices['pop']

    def updateIndices(self, priority:str='pop') -> None:
        '''Update indices
        (automatically reset pop and div after "pop x div" evaluations )

        Args:
            priority (str, optional): priority of loop. Defaults to 'pop'.
            - pop:
                div=0, pop=0 -> div=0, pop=1 -> div=0, pop=2 -> div=0, pop=3 ->...
            - div:
                div=0, pop=0 -> div=1, pop=0 -> div=2, pop=0 -> div=3, pop=0 ->...
        '''
        if priority == 'pop':
            self.indices['pop'] += 1
            if self.indices['pop'] >= self.cnf.max_pop:
                self.indices['div'] = (self.indices['div'] + 1) % self.max_div
                self.indices['pop'] = 0
        elif priority == 'div':
            self.indices['div'] += 1
            if self.indices['div'] >= self.max_div:
                self.indices['pop'] = (self.indices['pop'] + 1) % self.cnf.max_pop
                self.indices['div'] = 0

    def resetIndicesBy1cycle(self, priority:str='pop') -> bool:
        '''Reset indices

        Args:
            priority (str, optional): 1-cycle. Defaults to 'pop'.
            - pop:
                1 cycle: div=0, pop=0 ~ div=0, pop=max_pop-1
            - div:
                1 cycle: div=0, pop=0 ~ div=max_div-1, pop=0
        '''
        if priority == 'pop':
            if self.indices['div'] == 1:
                self.indices['div'], self.indices['pop'] = 0, 0
                return True
        elif priority == 'div':
            if self.indices['pop'] == 1:
                self.indices['div'], self.indices['pop'] = 0, 0
                return True
        return False


    def setCV(self, pop:Population) -> np.ndarray:
        '''Set context vector b*
        b* = ( pop.x[div][pop] | pop.x_best )

        Args:
            pop (Population): population with x[div][pop]

        Returns:
            np.ndarray: complete solution (prob_dim)
        '''
        _div, _pop = self.getIndices
        assert len(pop.x[_div][_pop])==self.dim[_div], f'Error: Sub-dimension does not match. ({self.__class__.__name__}.setCV)'
        b = pop.x_best.copy()
        subgroup = self.group[_div]
        b[subgroup] = pop.x[_div][_pop]
        return b

    def linkSolution(self, pop:Population) -> np.ndarray:
        '''Link solutions
        x = pop.x[:][pop]

        Args:
            pop (Population): population with x[0][pop] ~ x[max_div][pop]

        Returns:
            np.ndarray: complete solution (prob_dim)
        '''
        _pop = self.getIndices[1]
        x = np.full(self.fnc.prob_dim,np.nan)
        for _div,subgroup in enumerate(self.group):
            x[subgroup] = pop.x[_div][_pop]
        assert not any(pd.isnull(x))
        return x

    def assignFitness(self, pop:Population) -> Population:
        '''Assign fitness
        '''
        _div, _pop = self.getIndices
        if self.cnf.fitness_assignment == 'best':
            if self.superior(pop.f_new, pop.f[_div][_pop]):
                pop.f[_div][_pop] = pop.f_new
            else:
                pass
        elif self.cnf.fitness_assignment == 'current':
            pop.f[_div][_pop] = pop.f_new
        else:
            log(self, f'Error: Invalid fitness assignment ({self.cnf.fitness_assignment})', output=sys.stderr)
        return pop

    def SG(self) -> tuple[np.ndarray,np.ndarray]:
        '''Grouping by static grouping of m s-D subcomponents

        Note:
            m s-D static decomposition
                - m x s = [s,s,s,s,s,...,s]  <-- m subcomponents
            EX:
                1000D -> 20 50-D static decomposition
                1000D = [50D,50D,50D,...,50D]  <-- 20 subcomponents
                      = [[0,1,2,....,49],[50,51,...,99],....,[...,999]]
            Algorithm:
                CCPSO-S, CCPSO-Sk, etc.
        '''
        seps, nonseps = [], []
        # Generate subdim
        if (self.fnc.prob_dim % self.cnf.max_div) == 0 :
            subdims = [self.fnc.prob_dim//self.cnf.max_div] * self.cnf.max_div
        else :
            subdims = [self.fnc.prob_dim//self.cnf.max_div] * (self.cnf.max_div - 1)
            subdims.append( self.fnc.prob_dim - sum(subdims) )
        # Divide group
        dims = np.arange(self.fnc.prob_dim)
        total_subdim = 0
        for subdim in subdims:
            nonseps.append(dims[total_subdim:total_subdim+subdim])
            total_subdim += subdim
        return seps, nonseps

    def RG(self) -> tuple[np.ndarray,np.ndarray]:
        '''Grouping by random grouping with A fixed subcomponent size

        Note:
            Random Grouping with a fixed subcomponent size
                - m x s = [s,s,s,s,s,...,s]  <-- m subcomponents
                - s is randomly selected by all dimension index.
            EX:
                1000D -> 20 50-D static decomposition (randomly selected)
                1000D = [50D,50D,50D,...,50D]  <-- 20 subcomponents
                      = [[804,298,85,....,3],[629,413,...,912],....,[...,580]]
            Algorithm:
                DECC-I, DECC-G, CCPSO, etc.
        '''
        # set random variable
        if self.cnf.deterministic_grouping:
            self.rd = np.random
            self.rd.seed(0)
        else:
            self.rd = self.cnf.rd
        seps, nonseps = [], []
        # Generate subdim
        if (self.fnc.prob_dim % self.cnf.max_div) == 0 :
            subdims = [self.fnc.prob_dim//self.cnf.max_div] * self.cnf.max_div
        else :
            subdims = [self.fnc.prob_dim//self.cnf.max_div] * (self.cnf.max_div - 1)
            subdims.append( self.fnc.prob_dim - sum(subdims) )
        # Divide group
        dims = np.arange(self.fnc.prob_dim)
        self.rd.shuffle(dims)
        total_subdim = 0
        for subdim in subdims:
            nonseps.append(dims[total_subdim:total_subdim+subdim])
            total_subdim += subdim
        # reset random variable
        del self.rd
        if not self.cnf.deterministic_grouping:
            self.cnf.setRandomSeed(self.cnf.seed)
        return seps, nonseps

    def RDG3(self) -> tuple[np.ndarray,np.ndarray]:
        '''Grouping by RDG3 for overlapping problems

        Note:
            Recurse Differential Grouping 3
                - decomposition depending on variable interaction
            Algorithm:
                CC-RDG3, etc.

        Params:
            ep_n : threshold min element size
                - 50 (recommend):
                    robust parameter setting
                - 1000:
                    RDG3 == RDG2 (~RDG)
                    * RDG2 is adaptive parameter settings version of RDG.
        '''
        # set random variable
        if self.cnf.deterministic_grouping:
            self.rd = np.random
            self.rd.seed(0)
        else:
            self.rd = self.cnf.rd

        dim, axis_range = self.fnc.prob_dim, self.fnc.axis_range.copy()
        lb, ub = axis_range[:,0], axis_range[:,1]
        ep_n = 1000 if self.cnf.group_name == 'RDG2' else self.cnf.ep_n
        ep_s = self.cnf.ep_s

        seps, nonseps, S = [], [], set()
        x_ll = lb.copy()
        y_ll = self.getFitness(x_ll, True)
        x_remain = np.arange(dim)
        X1, X2 = set(x_remain[0:1]), set(x_remain[1:])
        x_remain = set(x_remain)
        # set object has '.remove' and '.disard' method
        # (list object doesn't have it)

        while len(x_remain) > 0:
            x_remain = set()
            _X1, x_remain = self.__getInteraction(X1, X2, x_remain, x_ll, y_ll, dim, lb, ub)
            if len(_X1) < ep_n and len(_X1) != len(X1):
                X1 = _X1.copy()
                X2 = x_remain.copy()
                if len(x_remain) == 0:
                    nonseps.append(np.array(list(X1)))
                    break
            else:
                if len(_X1) == 1:
                    S.add(list(_X1)[0])
                else:
                    nonseps.append(np.array(list(_X1)))
                if len(x_remain) > 1:
                    X1 = {list(x_remain)[0]}
                    x_remain.remove(list(X1)[0])
                    X2 = x_remain.copy()
                elif len(x_remain) == 1:
                    S.add(list(x_remain)[0])
                    break

        # convert S into seps
        while len(S) != 0:
            if len(S) < ep_s:
                seps.append(np.array(list(S)))
                S = set()
            else:
                _S = np.array(list(S)[:ep_s])
                seps.append(_S)
                for _s in _S:
                    S.remove(_s)

        # reset random variable
        del self.rd
        if not self.cnf.deterministic_grouping:
            self.cnf.setRandomSeed(self.cnf.seed)
        return seps, nonseps

    def __getInteraction(self,
        X1:set, X2:set, x_remain:set,
        x_ll:np.ndarray, y_ll:np.ndarray,
        dim:int,
        lb:np.ndarray, ub:np.ndarray
    )-> tuple[set,set]:
        '''Get variable interaction (for RDG3)

        Args:
            X1 (set): [description]
            X2 (set): [description]
            x_remain (set): [description]
            x_ll (np.ndarray): [description]
            y_ll (np.ndarray): [description]
            dim (int): [description]
            lb (np.ndarray): [description]
            ub (np.ndarray): [description]

        Returns:
            tuple[set,set]: [description]
        '''
        # machine dependence constant 2^(-52) in Python
        mdc = 2**(-52)
        # cast X1, X2 to ndarray
        _X1, _X2 = np.array(list(X1)), np.array(list(X2))

        # set position x_ul, x_lm, x_um
        x_ul = x_ll.copy()
        x_ul[_X1] = ub[_X1]
        x_lm, x_um = x_ll.copy(), x_ul.copy()
        x_lm[_X2],x_um[_X2]  = (lb[_X2] + ub[_X2])/2, (lb[_X2] + ub[_X2])/2

        # get fitness value y_ul, y_lm, y_um
        y_ul, y_lm, y_um = self.getFitness((x_ul, x_lm, x_um), True)

        # calculate fitness change delta1, delta2
        delta1, delta2 = (y_ll - y_ul), (y_lm - y_um)

        # estimate ep based on RDG2 (adaptive parameter)
        gamma = lambda k: k*mdc / (1.-k*mdc)
        ep = gamma(mt.sqrt(dim)+2.) * (abs(y_ll) + abs(y_ul) + abs(y_lm) + abs(y_um))

        if abs(delta1 - delta2) > ep:
            if len(X2) == 1:
                X1 = X1 | X2
            else:
                # random element
                self.rd.shuffle(_X2)
                mid = mt.floor(len(X2)/2)
                G1, G2 = set(_X2[:mid]), set(_X2[mid:])
                X1_1, x_remain = self.__getInteraction(X1, G1, x_remain, x_ll, y_ll, dim, lb, ub)
                X1_2, x_remain = self.__getInteraction(X1, G2, x_remain, x_ll, y_ll, dim, lb, ub)
                X1 = X1_1 | X1_2
        else:
            x_remain = x_remain | X2

        return X1, x_remain


class PSO(OptimizerCore):
    '''PSO (Particle Swarm Optimization)

    Note:
        - global best : use pop.x_best

    HowToUse:
        - Initialization
            - initializeParameter() and initializePopulation(pop)
        - Update
            - `updateParameter() and updatePopulation(pop)
    '''
    def __init__(self, cnf:Configuration, fnc:Function) -> None:
        '''
        Args:
            cnf (Configuration): Configuration class
            fnc (Function): Function class
        '''
        super().__init__(cnf, fnc)
        self.params:dict  = {}

    def initializeParameter(self) -> None:
        '''initialize parameter
        '''
        init_velocity:float = 0.
        self.params['velocity'] = [ np.full((self.cnf.max_pop, subdim), init_velocity) for subdim in self.dim ]
        self.params['x_pbest'] = [ np.full((self.cnf.max_pop, subdim), np.nan) for subdim in self.dim ]
        self.params['f_pbest'] = [ np.full(self.cnf.max_pop, self.init_fitness) for subdim in self.dim ]
        self.params['inertia'] = self.cnf.inertia
        self.params['accel_g'] = self.cnf.accel_g
        self.params['accel_p'] = self.cnf.accel_p


    def initializePopulation(self, pop:Population) -> Population:
        '''initialize all population
        '''
        return super().initializePopulation(pop)


    def updateParameter(self) -> None:
        '''update parameter
        '''
        pass


    def updatePopulation(self, pop:Population) -> Population:
        '''update population
        '''
        _div, _pop = self.getIndices
        self.params['velocity'][_div][_pop] = self.__updateVelocity(pop)
        pop.x[_div][_pop] = self.__updatePosition(pop)
        pop.x_new = self.setCV(pop)
        pop.f_new = self.getFitness(pop.x_new)
        pop = self.updateBest(pop, pop.x_new, pop.f_new)
        self.updateIndices('pop')
        return pop


    def updateBest(self, pop:Population, x_new:np.ndarray, f_new:float) -> Population:
        '''Update global and personal best position and fitness value
        '''
        _div, _pop = self.getIndices
        subgroup = self.group[_div]
        # global best
        if self.superior(f_new, pop.f_best):
            pop.x_best = x_new.copy()
            pop.f_best = f_new
        # personal best
        if self.superior(f_new, self.params['f_pbest'][_div][_pop]):
            self.params['x_pbest'][_div][_pop] = x_new[subgroup]
            self.params['f_pbest'][_div][_pop] = f_new
        return pop


    def __updateVelocity(self, pop:Population) -> np.ndarray:
        '''Update velocity
        '''
        _div, _pop = self.getIndices
        subgroup, subdim = self.group[_div], self.dim[_div]
        # update velocity
        v = self.params['inertia']*self.params['velocity'][_div][_pop]
        + self.params['accel_g']*self.cnf.rd.rand(subdim)*(pop.x_best[subgroup]-pop.x[_div][_pop])
        + self.params['accel_p']*self.cnf.rd.rand(subdim)*(self.params['x_pbest'][_div][_pop]-pop.x[_div][_pop])
        return v


    def __updatePosition(self, pop:Population) -> np.ndarray:
        '''Update position by velocity
        '''
        _div, _pop = self.getIndices
        subgroup, subdim = self.group[_div], self.dim[_div]
        lb, ub = self.fnc.axis_range[subgroup,0], self.fnc.axis_range[subgroup,1]
        # update position
        x = pop.x[_div][_pop] + self.params['velocity'][_div][_pop]
        # boundary treatment
        x = np.clip( x, lb, ub )
        return x


