# Sub Optimizer
# version 1.1 (2022/01/12)

# standard library
import sys
import copy             as cp
import math             as mt
# additional library
import numpy            as np
from utils              import log,Stdio
''' For Annotations '''
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


class OptimizerCore:
    '''Core optimizer method
    '''
    def __init__(self, cnf:Configuration, fnc:Function) -> None:
        self.cnf    = cnf
        self.fnc    = fnc
        # update index
        self.indices = { 'div': 0, 'pop': 0 }
        # group
        self.group = []
        self.subdim = []


    '''
        Optimization (Minimize & Maximize)
    '''
    def setOptType(self, fnc_type:str) -> None:
        '''Guarantee optimization (minimization/maximization) judgment
        '''
        max_prob = []       # maximization problem lists
        if fnc_type in max_prob:
            self.cnf.opt_type = 'max'
        else:
            self.cnf.opt_type = 'min'


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


    def getFitness(self, x:list|tuple|np.ndarray) -> tuple|float :
        '''get single/multi fitness value

        Args:
            x (list or tuple or np.ndarray): single/multi solution
                single solution:
                    x : ndarray(dim)
                multi solutions:
                    xs : [x,x,...,x] or (x,x,...,x) or ndarray(x,x,..,x)
                    * x type is ndarray.

        Returns:
            float : fitness value of single solution
            tuple : fitness values of multi solution
        '''
        # x is composed of multi decision value
        if (isinstance(x, np.ndarray) and x.ndim ==2 ) or isinstance(x,(list,tuple)):
            if self.fnc.total_evals >= self.cnf.max_evals:
                return 'Exceed FEs'
            return tuple(map(self.fnc.doEvaluate,x))
        # x is composed of single decision value
        elif isinstance(x, np.ndarray) and x.ndim ==1:
            return self.fnc.doEvaluate(x)
        else:
            log(self, f'Error: Invalid type argument in getFitness(). {type(x)}', output=sys.stderr)
            return 'Invalid Input'


    def initializePopulation(self, pop:Population) -> Population:
        if self.cnf.opt_type == 'min':
            init_fitness = np.inf
        elif self.cnf.opt_type == 'max':
            init_fitness = -np.inf
        # set variables
        pop_size, dim, axis_range = self.cnf.max_pop, self.fnc.prob_dim, self.fnc.axis_range.copy()
        lb, ub = axis_range[:,0], axis_range[:,1]
        # initialize population
        assert dim == axis_range.shape[0], 'Error: Dimension does not match.'
        pop.x = self.cnf.rd.uniform(lb, ub, (pop_size, dim))
        pop.f = np.full((pop_size, self.div), init_fitness)
        pop.x_best = np.full(dim,np.nan)
        pop.f_best = init_fitness
        return pop


    def updateBest(self, pop:Population, x_new:list|np.ndarray, f_new:float) -> Population:
        '''
            update best 3 position and fitness
        '''
        if self.superior(f_new, pop.f_best):
            pop.x_best = x_new.copy()
            pop.f_best = f_new
        return pop

    def getBestIndices(self, fs:list|np.ndarray, n_output:int=1) -> float|np.ndarray:
        '''Get best index(or indices) of the fitness values
        '''
        assert isinstance(n_output,int) and 1<=n_output<=len(fs), f'Error: n_output {n_output} is invalid value'
        if self.cnf.opt_type == 'min':
            return np.argsort(fs)[0] if n_output==1 else np.argsort(fs)[:n_output][::-1]
        elif self.cnf.opt_type == 'max':
            return np.argsort(fs)[-1] if n_output==1 else np.argsort(fs)[-n_output:][::-1]
        else:
            log(self, f'Error: Invalid opt_type "{self.cnf.opt_type}".', output=sys.stderr)
            sys.exit(1)

    '''
        Cooperative Co-evolution
    '''
    def updateIndices(self) -> None:
        '''update indices of individual
        '''
        self.indices['pop'] += 1
        if self.indices['pop'] >= self.cnf.max_pop:
            self.indices['div'] = (self.indices['div'] + 1) % self.max_div
            self.indices['pop'] = 0

    def assignFitness(self, pop: Population) -> Population:
        '''fitness assignment
        '''
        _pop = self.indices['pop']
        if self.cnf.fitness_assignment == 'best':
            if self.superior(pop.f_new, pop.f[_pop]):
                pop.f[_pop] = pop.f_new
            else:
                pass
        elif self.cnf.fitness_assignment == 'current':
            pop.f[_pop] = pop.f_new
        else:
            log(self, f'Error: Invalid fitness assignment ({self.cnf.fitness_assignment})', output=sys.stderr)
        return pop

    def updateIndices(self) -> None:
        '''update indices of individual
        '''
        self.indices['pop'] += 1
        if self.indices['pop'] >= self.cnf.max_pop:
            self.indices['div'] = (self.indices['div'] + 1) % self.max_div
            self.indices['pop'] = 0

    def SG(self) -> tuple[np.ndarray,np.ndarray]:
        '''Grouping by static grouping
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

    def RDG3(self) -> tuple[np.ndarray,np.ndarray] :
        '''Grouping by RDG3 for overlapping problems

        Note:
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
        y_ll = self.getFitness(x_ll)
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
        y_ul, y_lm, y_um = self.getFitness((x_ul, x_lm, x_um))

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
        self.params  = {}

    def initializeParameter(self) -> None:
        '''initialize parameter
        '''
        self.params['crossover_rate'] = self.cnf.crossover_rate
        self.params['mutation_rate'] = self.cnf.mutation_rate


    def initializePopulation(self, pop:Population) -> Population:
        '''initialize all population
        '''
        return super().initializePopulation(pop)


    def updateParameter(self) -> None:
        '''update parameter
        '''
        pass


    def updatePopulation(self, pop:Population) -> Population:
        '''update position
        '''
        div = self.indices['div']
        _pop = self.indices['pop']
        l = self.indices['div']+1

        if len(self.params['ls_queue'][l]) == 0:
            # context vector
            x_new = cp.deepcopy(pop.x_best)

            # [1] elite selection
            xs = [pop.x[i][div]  for i in range(len(pop.x))]
            x1, x2 = xs[_pop].copy(), xs[self.getBestIndices(pop.f)].copy()
            # TODO: ベスト２個体以外が有効利用されない可能性がある

            # [2] crossover
            if self.cnf.rd.uniform(0,1) < self.params['crossover_rate'] :
                if self.cnf.abbrev_solution:
                    # 2-point crossover
                    point = np.random.choice(len(x1), 2, replace=False)
                    point.sort()
                    if self.cnf.rd.uniform(0,1) < 0.5:
                        x_layer_new = x1[:point[0]]
                        x_layer_new.extend(x2[point[0]:point[1]+1])
                        x_layer_new.extend(x1[point[1]+1:])
                    else:
                        x_layer_new = x2[:point[0]]
                        x_layer_new.extend(x1[point[0]:point[1]+1])
                        x_layer_new.extend(x2[point[1]+1:])
                else:
                    # orderd crossover (OX)
                    # (1) select cross point
                    point = np.random.choice(len(x1), 2, replace=False)
                    point.sort()
                    s1 = x1[point[0]:point[1]+1]
                    # (2) copy to x_new
                    x_layer_new = [None]*len(x1)
                    x_layer_new[point[0]:point[1]+1] = s1
                    # (3) remove the elements
                    s2 = x2.copy()
                    for element in s1:
                        s2.remove(element)
                    # (4) copy to x_new
                    s2_point = 0
                    for i,x_element in enumerate(x_layer_new):
                        if x_element is None:
                            x_layer_new[i] = s2[s2_point]
                            s2_point+=1
                    assert s2_point==len(s2), 'Error: fail to update position in crossover.'
            else:
                x_layer_new = x1.copy()

            # [3] mutation
            if self.cnf.rd.uniform(0,1) < self.params['mutation_rate'] :
                if self.cnf.abbrev_solution:
                    # all random mutation (replacement)
                    for _index,x_element in enumerate(self.group[self.indices['div']]):
                        x_layer_new[_index] = self.cnf.rd.choice(self.cell_size[self.getLayer(x_element)])
                else:
                    # reverse mutation (replacement)
                    x_layer_new = x_layer_new[::-1]
            assert len(x_layer_new)==len(x1), 'Error: change solution size by inappropriate operation.'

            # create perfect solution
            if self.cnf.abbrev_solution:
                for _index in range(len(self.group[div])):
                    x_new = self.swapCC(x_new, (div,_index), x_layer_new[_index])
                x_new = self.clip(x_new)
            else:
                x_new[div] = x_layer_new.copy()

            # evaluate x
            returns = self.getFitness(x_new)
            if returns == 'No Fitness':
                # no fitness -> pass
                return pop
            pop.f_new, pop.f_org_new, pop.NoVW_new, pop.NoVL_new, pop.totalWL_new, pop.calc_time = returns
            # !!CAUTION!! pop.x[_pop] is NOT best array
            pop.x[_pop][div] = x_new[div].copy()
            # assign fitness
            pop = self.assignFitness(pop)
            # update best individual
            pop = self.updateBest(pop, x_new, pop.f_new)

        if self.cnf.ls_name != None:
            pop = self.updateByLocalSearch(pop)

        return pop


