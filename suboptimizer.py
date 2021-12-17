# Sub Optimizer
# version 2.1 (2021/11/30)
# for Optimizer >= v5.0

import sys                              # standard library
import re                               # standard library
import copy             as cp           # standard library
from typing import Callable, List, Tuple, Union      # standard library
import math             as mt
import numpy            as np
import pandas           as pd
import itertools        as it

from logger import Stdio
''' For Function Annotations '''
import config           as cf
import function         as fc

cnf_glb = cf.Configuration()


class Population:
    '''
        population class for optimizer
    '''
    def __init__(self) -> None:
        '''
            initialization of all variables
        '''
        self.x      = []
        self.f      = np.inf
        # current log
        self.f_new     = np.inf
        self.f_org_new = -np.inf
        self.NoVW_new  = np.inf
        self.NoVL_new  = np.inf
        self.totalWL_new = np.inf
        self.calc_time = 0
        # best log
        self.x_best     = []
        self.x_best_out = []        # cell array (without flag)
        self.f_best     = np.inf
        self.f_org_best = -np.inf
        self.NoVW_best  = np.inf
        self.NoVL_best  = np.inf
        self.totalWL_best = np.inf


    @property
    def getBest(self) -> tuple:
        """
            best getter method (external reference)
        """
        return self.x_best_out, self.f_best, self.f_org_best, self.NoVW_best, self.NoVL_best, self.totalWL_best

    @property
    def getCurrent(self) -> tuple:
        """
            current getter method (external reference)
        """
        return self.f_new, self.f_org_new, self.NoVW_new, self.NoVL_new, self.totalWL_new, self.calc_time


class Core:
    '''
        ## Core optimizer method
    '''
    def __init__(self, cnf:cf.Configuration, fnc:fc.Function) -> None:
        self.cnf    = cnf
        self.fnc    = fnc
        # update index
        self.indices = { 'div': 0, 'pop': 0 }
        # regular expresion : [str]_[str]_[str]
        self.pattern = '(.*?)_(.*)'
        self.prog = re.compile(self.pattern)
        # group
        self.group = []


    '''
        Optimization (Minimize & Maximize)
    '''
    def setOptType(self, fnc_type:str) -> None:
        '''
            Guarantee optimization (minimization/maximization) judgment

            Attributes
            ----------
            fnc_type: str
        '''
        max_prob = ['original']

        if fnc_type in max_prob:
            self.cnf.opt_type = 'max'
        else:
            self.cnf.opt_type = 'min'


    def superior(self, value_1:float, value_2:float) -> bool:
        '''
            whether value_1 is superior to value_2
            (if value_1 is equal to value_2, function returns False)
        '''
        # TODO: Debug Error Occur
        # assert isinstance(value_1, float) and isinstance(value_2, float), 'Error: invalid type function in superior'
        if not isinstance(value_1, float):
            print(f'[OptimizerCore] Fatal Error: Invalid type function in superior value_1 = {value_1} (class {self.__class__.__name__}).', file=sys.stderr)
            return False
        elif not isinstance(value_2, float):
            print(f'[OptimizerCore] Fatal Error: Invalid type function in superior value_2 = {value_2} (class {self.__class__.__name__}).', file=sys.stderr)
            return False
        if self.cnf.opt_type == 'min':
            return True if value_1 < value_2 else False
        elif self.cnf.opt_type == 'max':
            return True if value_1 > value_2 else False
        else:
            print(f'[OptimizerCore] Error: Invalid opt_type "{self.cnf.opt_type}" (class {self.__class__.__name__}).', file=sys.stderr)
            sys.exit(1)


    def getFitness(self, x:list, counter:int=0) -> tuple :
        '''
            get single/multi fitness value
            * You can also get fitness when x is the following type:
                - tuple-ndarray : (ndarray, ..., ndarray)
                - list-ndarray  : [ndarray, ..., ndarray]
                - list-list  : [list, ..., list]
        '''
        if self.cnf.abbrev_solution:
            x = self.decodeCC(x)
        assert len(x)==self.fnc.layers, 'Error: The number of layers does not match.'
        x_decode = self.decode(x)
        # x_decode = cp.deepcopy(x)
        x_depth = self.getDepth(x_decode)
        # x is composed of single decision value
        if x_depth == 2:
            if self.fnc.total_evals >= self.cnf.max_evals:
                return 'Exceed FEs'
            return self.fnc.doEvaluate(x_decode)
        else:
            print(f'[OptimizerCore] Error: The number of layers {x_depth} does not match.', file=sys.stderr)
            return None


    '''
        Blank Optimization
    '''
    def blankOpt(self, total_evals:int, init_evals:int=0) -> bool:
        '''
            blank optimization timing

            Attributes
            ----------
            total_evals : int
        '''
        if self.cnf.blank_opt == 'on':
            return True
        elif self.cnf.blank_opt == 'off':
            return False
        elif self.cnf.blank_opt == '2-stage':
            return False  if (total_evals-init_evals) < self.cnf.switch_timing * (self.cnf.max_evals-init_evals)  else  True
        else:
            raise ValueError(f'Invalid cnf.blank_opt {self.cnf.blank_opt}')


    def fillPositionwithBlank(self, pop:Population) -> Population:
        '''
            fill positions with blanks from behind
            [ x1, x2, x3 ] -> [ x1, x2, x3, blank_string, ..., blank_string ]
        '''
        # fill population
        for i,x in enumerate(pop.x):
            if self.cnf.abbrev_solution:
                for j,x_div in enumerate(x):
                    x_new = cp.deepcopy(pop.x_best)
                    x_new[j] = x_div
                    x_dec = self.decode(self.decodeCC(x_new))
                    for layer in range(len(x_dec)):
                        x_dec[layer].extend( [self.fnc.blank_string]*self.fnc.max_blanks[layer+1] )
                        if i==0 and j==0:
                            self.cell_size[layer+1] += self.fnc.max_blanks[layer+1]
                    pop.x[i][j] = self.encodeCC(self.encode(x_dec))[j]
            else:
                x_dec = self.decode(x)
                for layer in range(self.fnc.layers):
                    x_dec[layer].extend( [self.fnc.blank_string]*self.fnc.max_blanks[layer+1] )
                    if i==0:
                        self.cell_size[layer+1] += self.fnc.max_blanks[layer+1]
                pop.x[i] = self.encode(x_dec)
        # fill x_best
        x_best_dec = self.decode(self.decodeCC(pop.x_best))  if self.cnf.abbrev_solution  else self.decode(pop.x_best)
        for layer in range(self.fnc.layers):
            x_best_dec[layer].extend( [self.fnc.blank_string]*self.fnc.max_blanks[layer+1] )
        pop.x_best = self.encodeCC(self.encode(x_best_dec))  if self.cnf.abbrev_solution  else self.encode(x_best_dec)
        self.blank_opt = True
        return pop

    '''
        Solution Expression List

                             decodeCC                 decode               evaluate     
        ---------------------       ------------------       --------------             
        |                   | ----> |                | ----> |            |             
        | cc position array |       | position array |       | cell array | ----> ( f ) 
        |                   | <---- |                | <---- |            |             
        ---------------------       ------------------       --------------             
                             encodeCC                 encode                            
    '''
    def encode(self, x_cell:list) -> list:
        '''
            encode cell array to position array

            Attributes
            ----------
            x : list[ list, list, ..., list ]
                cell array

            Returns
            -------
            x_encode : list[ list, list, ..., list ]
                position array
        '''
        x_encode = []
        for layer,x_layer in enumerate(x_cell,start=1):
            domain = self.getDomain(layer=layer)
            # type casting
            # x_layer = x_layer if isinstance(x_layer, list) else list(x_layer)
            _x_layer = [ x_layer.index(cell)  for cell in domain ]
            if len(x_layer) > len(domain):
                _x_layer.extend([ k for k, v in enumerate(x_layer)  if v == self.fnc.blank_string ])
            x_encode.append(_x_layer)
        return x_encode


    def decode(self, x_pos:list) -> list:
        '''
            decode position array to cell list

            Attributes
            ----------
            x : list[ list, list, ..., list ]
                position list

            Returns
            -------
            x_encode : list[ list, list, ..., list ]
                cell list
        '''
        x_decode = []
        for layer,x_layer in enumerate(x_pos,start=1):
            domain = self.getDomain(layer=layer)
            dimension = len(x_layer)
            x_layer_decode = [None]*dimension
            for _index,x_element in enumerate(x_layer):
                x_layer_decode[x_element] = domain[_index]  if _index < len(domain) else self.fnc.blank_string
            # type casting
            x_decode.append(x_layer_decode)
        return x_decode


    def encodeCC(self, x_pos:list) -> list:
        '''
            encode position array to cc position array

            Attributes
            ----------
            x : list[ list, list, ..., list ]
                cell array

            Returns
            -------
            x_encode : list[ list, list, ..., list ]
                position array
        '''
        assert len(self.group)>0, f'Error: invalid group {self.group}'
        x_decode = [ [None]*len(cell_layer)  for cell_layer in self.group ]
        # cc side loop
        for _layer,cc_layer in enumerate(self.group):
            for _index,cc_element in enumerate(cc_layer):
                row,col = self.getIndex(list(self.fnc.domain.values()),cc_element)
                x_decode[_layer][_index] = x_pos[row][col]
        return x_decode


    def decodeCC(self, x_ccpos:list, cc_layer:int=-1, x_ccpos_cv:list=[]) -> list:
        '''
            decode cc position array to position array

            Attributes
            ----------
            x_ccpos : list or list[ list, list, ..., list ]
                cc position array (1-d or 2-d array)

            cc_layer : int
                layer number in cc (1 ~ layers)

            x_ccpos_cv : list or list[ list, list, ..., list ]
                cc position context vector array (0-d or 2-d array)

            Returns
            -------
            x_encode : list[ list, list, ..., list ]
                position array
        '''
        assert len(self.group)>0, f'Error: invalid group {self.group}'
        # set context vector
        cv_depth = self.getDepth(x_ccpos_cv)
        if cv_depth == 0:
            x_encode = cp.deepcopy(x_ccpos)
            for i in range(self.fnc.layers):
                if len(x_encode[i]) < self.cell_size[i+1]:
                    x_encode[i].extend([None]*(self.cell_size[i+1]-len(x_encode[i])))
        elif cv_depth == 2:
            x_encode = cp.deepcopy(self.encode(x_ccpos_cv))
        else:
            raise AttributeError('invalid x_ccpos_cv in function')

        ccpos_depth = self.getDepth(x_ccpos)
        # 1-d cell array
        if ccpos_depth == 1:
            assert cv_depth==0, f'Error: invalid x_ccpos in {sys._getframe().f_code.co_name}'
            assert cc_layer==-1, f'Error: invalid cc_layer {cc_layer} in {sys._getframe().f_code.co_name}'
            x_encode[cc_layer] = x_ccpos.copy()
        # 2-d cell array
        elif ccpos_depth == 2:
            for _layer,cell_layer in enumerate(self.group,start=1):
                for _index,cell_element in enumerate(cell_layer):
                    cell_layer = self.getLayer(cell_element)
                    domain = self.getDomain(layer=cell_layer)
                    domain_index = domain.index(cell_element)
                    x_encode[cell_layer-1][domain_index] = x_ccpos[_layer-1][_index]
        indices = self.getIndex(x_encode, None, 'multi')
        # complement blank
        if len(indices)!=0:
            for (row,col) in indices:
                x_remain = set(range(len(x_encode[row]))) - set(x_encode[row])
                x_encode[row][col] = list(x_remain)[0]
        return x_encode


    def swap(self, x_layer:list, index1:int, num2:int):
        '''
            swap index1(num1) and index2(num2)
        '''
        assert isinstance(x_layer, list), f'Error: x_layer is invalid type {type(x_layer)}, not list.'
        if num2 in x_layer:
            index2 = x_layer.index(num2)
        else:
            print(f'[OptimizerCore] Error: {num2} does not found in list {x_layer}.', file=sys.stderr)
            sys.exit(1)
        if index1 != index2:
            x_layer[index1], x_layer[index2] = num2, x_layer[index1]
        return x_layer


    def swapCC(self, x_ccpos:list, index1:tuple, num2:int) -> list:
        '''
            swap index1(num1) and index2(num2)
        '''
        assert len(self.group)>0, f'Error: invalid group {self.group}'
        x_swap = cp.deepcopy(x_ccpos)
        row1,col1 = index1
        cellname = self.group[row1][col1]
        layer = self.getLayer(cellname)
        pos_remain = list(range(self.cell_size[layer]))
        # simple swap = swap()
        if num2 in x_swap[layer-1]:
            row2,col2 = layer-1,x_swap[layer-1].index(num2)
            x_swap[row1][col1], x_swap[row2][col2] = num2, x_swap[row1][col1]
        else:
            # remain check (specify no use cell)
            pos_remain = list(set(pos_remain) - set(x_swap[layer-1]))
            # duplicate check (specify 2 more times use cell)
            dup_element = [ element for element in set(x_swap[layer-1]) if x_swap[layer-1].count(element) > 1 ]
            # random select from other elements
            replace_targets = list(range(len(x_swap[layer-1])))
            replace_targets.remove(col1)        # other elements, not self
            row2,col2 = layer-1, self.cnf.rd.choice(replace_targets)
            x_swap[row1][col1], x_swap[row2][col2] = num2, self.cnf.rd.choice(pos_remain+dup_element)
        return x_swap


    def updateIndices(self) -> None:
        '''
            update indices of individual
        '''
        self.indices['pop'] += 1
        if self.indices['pop'] >= self.cnf.max_pop:
            self.indices['div'] = (self.indices['div'] + 1) % self.max_div
            self.indices['pop'] = 0


    def updateBest(self, pop:Callable, x_new:Union[list,np.ndarray], f_new:float) -> Population:
        '''
            update best 3 position and fitness
        '''
        if self.superior(f_new, pop.f_best):
            pop.x_best = cp.deepcopy(x_new)
            pop.f_best = f_new
            pop.f_org_best = pop.f_org_new
            pop.x_best, pop.f_best = cp.deepcopy(x_new), f_new
            pop.NoVW_best = pop.NoVW_new
            pop.NoVL_best = pop.NoVL_new
            pop.totalWL_best = pop.totalWL_new
            if self.cnf.abbrev_solution:
                pop.x_best_out = self.fnc.removeFlag(self.decode(self.decodeCC(pop.x_best)))
            else:
                pop.x_best_out = self.fnc.removeFlag(self.decode(pop.x_best))
        return pop


    def clip(self, x_ccpos:list, div:int=-1) -> list:
        '''
            clip values in domain (for abbrev expression)
        '''
        x_pos = self.decodeCC(x_ccpos)
        if div != -1:
            # fix div layer position
            layer = div
            x_layer_component = set(x_pos[layer])
            layer_component = set(range(self.cell_size[layer+1]))
            # unusual components
            if len(x_layer_component ^ layer_component) != 0:
                # domain check
                if len(x_layer_component - layer_component) != 0:
                    vio_pos = list(x_layer_component - layer_component)
                    cor_pos = list(layer_component - x_layer_component)
                    for _from,_to in zip(vio_pos, cor_pos):
                        x_pos[layer][x_pos[layer].index(_from)] = _to
                    x_layer_component = set(x_pos[layer])
                # duplicate check
                if len(layer_component) != len(x_layer_component):
                    dup_pos = list(layer_component - x_layer_component)
                    for cell in x_layer_component:
                        cnt = x_pos[layer].count(cell)
                        if cnt >= 2:
                            for _cnt in range(cnt-1):
                                x_pos[layer][x_pos[layer].index(cell)] = dup_pos[0]
                                dup_pos.pop(0)
                                if len(dup_pos) == 0:
                                    break
                            else:
                                continue
                            break
                    x_layer_component = set(x_pos[layer])
                assert len(x_layer_component ^ layer_component) == 0, 'Error: mistake clip position'
        else:
            # fix position in position array
            for layer,x_layer in enumerate(x_pos):
                x_layer_component = set(x_layer)
                layer_component = set(range(self.cell_size[layer+1]))
                # unusual components
                if len(x_layer_component ^ layer_component) != 0:
                    # domain check
                    if len(x_layer_component - layer_component) != 0:
                        vio_pos = list(x_layer_component - layer_component)
                        cor_pos = list(layer_component - x_layer_component)
                        for _from,_to in zip(vio_pos, cor_pos):
                            x_pos[layer][x_pos[layer].index(_from)] = _to
                        x_layer_component = set(x_pos[layer])
                    # duplicate check
                    if len(layer_component) != len(x_layer_component):
                        dup_pos = list(layer_component - x_layer_component)
                        for cell in x_layer_component:
                            cnt = x_layer.count(cell)
                            if cnt >= 2:
                                for _cnt in range(cnt-1):
                                    x_pos[layer][x_pos[layer].index(cell)] = dup_pos[0]
                                    dup_pos.pop(0)
                                    if len(dup_pos) == 0:
                                        break
                                else:
                                    continue
                                break
                        x_layer_component = set(x_pos[layer])
                    assert len(x_layer_component ^ layer_component) == 0, 'Error: mistake clip position'
        return self.encodeCC(x_pos)


    def getLayer(self, x_element:str) -> int:
        return int(self.prog.match(x_element).group(1))


    def getIndex(self, _list:list, search_string:Union[str,None], output_type:str='single') -> Union[tuple,list]:
        '''
            get index of 2d array
        '''
        pos = []
        for row,_list_layer in enumerate(_list):
            for col, element in enumerate(_list_layer):
                if _list[row][col] == search_string:
                    pos.append((row, col))
                    if output_type == 'single':
                        return pos[0]
        return pos


    def getDomain(self, x_element:str='none', layer:int=0) -> list:
        '''
            get domain from x_element, layer (e.g. '1_in_2' )
        '''
        if not x_element == 'none':
            layer = self.getLayer(x_element)
        return self.fnc.domain[layer]


    def getDepth(self, _list:list) -> int:
        if not isinstance(_list, (list, np.ndarray)) or _list==[]:
            return 0
        else:
            if isinstance(_list, (list, np.ndarray)):
                return 1 + max(self.getDepth(element) for element in _list)
            else:
                return 0


    def getBestIndices(self, fs:Union[list, np.ndarray], n_output:int=1) -> Union[float, np.ndarray]:
        '''
            get best index(or indices) of the fitness values
        '''
        assert isinstance(n_output,int) and 1<=n_output<=len(fs), f'Error: n_output {n_output} is invalid value'
        if self.cnf.opt_type == 'min':
            return np.argsort(fs)[0] if n_output==1 else np.argsort(fs)[:n_output][::-1]
        elif self.cnf.opt_type == 'max':
            return np.argsort(fs)[-1] if n_output==1 else np.argsort(fs)[-n_output:][::-1]
        else:
            print(f'[OptimizerCore] Error: Invalid opt_type "{self.cnf.opt_type}" (class {self.__class__.__name__}).', file=sys.stderr)
            sys.exit(1)


    def assignFitness(self, pop: Population) -> Population:
        '''
            fitness assignment
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
            print('[OptimizerCore] Error: Invalid fitness assignment ({})'.format(self.cnf.fitness_assignment), file=sys.stderr)
        return pop


    def updateByLocalSearch(self, pop:Population) -> Population:
        div = self.indices['div']
        _pop = self.indices['pop']
        l = self.indices['div']+1
        # context vector
        x_org = self.decode(pop.x[_pop])[div]
        x_best = cp.deepcopy(self.decode(pop.x_best))
        # 2-opt/3-opt
        if self.cnf.ls_name == '2-opt':
            next_edge = lambda k:k+1 if k+1<len(x_org) else 0
            # generate only queue
            if len(self.params['ls_queue'][l]) == 0:
                for cells in it.combinations(x_org,2):
                    self.params['ls_queue'][l].append(cells)
            # get current cell-pairs from queue and local search
            else:
                (cell1,cell2) = self.params['ls_queue'][l][0]
                x_cand = x_org.copy()
                i_blank = len(self.fnc.domain[l])
                # edge1 = (edge1_1,edge1_2), edge2 = (edge2_1,edge2_2)
                edge1_1, i_blank = self.getIndexOfCells(cell1,l,i_blank)
                edge1_2 = next_edge(edge1_1)
                edge2_1, i_blank = self.getIndexOfCells(cell2,l,i_blank)
                # edge2_2 = next_edge(edge2_1)
                if not edge1_2 == edge2_1:
                    x1 = x_org[edge1_2:edge2_1+1]
                    # reverse x1
                    x_cand[edge1_2:edge2_1+1] = x1[::-1]
                    x_best[div] = x_cand.copy()
                    x_new = self.encode(x_best)
                    # evaluate x
                    self.params['ls_count'] += 1
                    returns = self.getFitness(x_new)
                    if returns in ['No Fitness', 'Exceed FEs']:
                        # no fitness -> pass
                        print(f'[Optimizer] Function Return is invalid {returns}')
                        return pop
                    pop.f_new, pop.f_org_new, pop.NoVW_new, pop.NoVL_new, pop.totalWL_new, pop.calc_time = returns
                    # update best individual
                    pop = self.updateBest(pop, x_new, pop.f_new)
                # x1 size == 1 (edge1_2==edge2_1) -> pass
                else:
                    pass
                # remove queue
                self.params['ls_queue'][l].pop(0)
        elif self.cnf.ls_name == '3-opt':
            print('[Optimizer] "3-opt" is not implemented yet!')
            sys.exit(1)
        else:
            print(f'[Optimizer] Invalid ls_name {self.cnf.ls_name}', file=sys.stderr)
        return pop


    def getIndexOfCells(self, cellname:str, l:int, i_blank:int):
        '''get index of cells

        Args:
            cellname (str): cellname (including blank)
            l (int): layer number
            i_blank (int): blank counter

        Raises:
            ValueError: invalid cellname

        Returns:
            i(int), i_blank(int): number, blank counter
        '''
        if cellname in self.fnc.domain[l]:
            return self.fnc.domain[l].index(cellname),i_blank
        elif cellname == self.fnc.blank_string:
            return i_blank, i_blank+1 if i_blank+1<self.cell_size[l] else len(self.fnc.domain[l])
        else:
            raise ValueError(f'[Optimizer] Invalid x_layer_new "{cellname}"')


class PSO(Core):
    '''
        ## GA (Genetic Algorithm) for TSP

        Attributes
        ----------
        `cnf` : Configuration class

        `fnc` : Function class

        Using Method
        ------------
        - Initialization
            - `initializeParameter()` and `initializePopulation(pop)`
        - Update
            - `updateParameter()` and `updatePopulation(pop)`
    '''
    def __init__(self, cnf:cf.Configuration, fnc:fc.Function) -> None:
        super().__init__(cnf, fnc)
        self.params  = {}


    def initializeParameter(self) -> None:
        '''
            initialize parameter
        '''
        self.params['crossover_rate'] = self.cnf.crossover_rate
        self.params['mutation_rate'] = self.cnf.mutation_rate
        self.params['ls_count'] = 0
        self.params['ls_queue'] = { layer: [] for layer in self.fnc.cell_size}


    def initializePopulation(self, pop:Population) -> None:
        '''
            initialize all population
        '''
        # set variables
        pop_size, random = self.cnf.max_pop, self.cnf.rd
        # initialize population elements
        pop.x = [
            self.encode(
                [ random.permutation(_x_phase).tolist()  for _x_phase in self.fnc.domain.values() ])
                for _pop in range(pop_size)
            ]
        if self.cnf.opt_type == 'min':
            init_fitness = np.inf
        elif self.cnf.opt_type == 'max':
            init_fitness = -np.inf
        pop.x_best = self.encode(
            [ random.permutation(_x_phase).tolist()  for _x_phase in self.fnc.domain.values() ]
        )
        pop.f = np.full(pop_size, init_fitness)
        pop.f_best = init_fitness
        return pop


    def updateParameter(self) -> None:
        '''
            update parameter
        '''
        pass


    def updatePopulation(self, pop:Population) -> Population:
        '''
            update position
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

