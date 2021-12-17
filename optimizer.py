# Optimizer
# version 1.0 (2021/12/17)

import os                               # standard library
import sys                              # standard library
import math             as mt
import numpy            as np
import pandas           as pd

from logger import Stdio
''' For Function Annotations '''
import config           as cf
import function         as fc
from suboptimizer       import *


class CCRDG3(PSO):
    '''
        ## CC-RDG3

        Attributes
        ----------
        `cnf` : Configuration class

        `fnc` : Function class

        Using Method
        ------------
        - Main execution
            - `initialize(blank_opt)` -->  `update(blank_opt)` (repeat for termination)
    '''
    def __init__(self, cnf:cf.Configuration, fnc:fc.Function) -> None:
        super().__init__(cnf, fnc)
        self.pop = Population()
        # static grouping
        self.max_div = self.fnc.layers
        self.blank_opt = False
        self.init_evals = 0
        # cell size
        self.cell_size = self.fnc.cell_size.copy()


    def initialize(self, blank_opt:bool=True) -> None:
        '''
            initialize method (external reference)
            (one-call -> one-evaluation)
        '''
        _pop = self.indices['pop']
        if _pop == 0:
            # set opt type
            self.setOptType(self.cnf.function_type)
            # grouping method : None
            seps, nonseps = self.grouping()
            self.group = [seps.copy()]  if seps != [] else []
            if self.cnf.nonseps_div:
                nonseps = nonseps if isinstance(nonseps, list) else [nonseps]
                # max_groupsize division
                for _nonsep in nonseps:
                    if len(_nonsep) > self.cnf.max_groupsize:
                        loop = mt.ceil(len(_nonsep)/self.cnf.max_groupsize)
                        for _loop in range(loop):
                            self.group.append(_nonsep[_loop*self.cnf.max_groupsize:(_loop+1)*self.cnf.max_groupsize-1])
                    else:
                        self.group.append(_nonsep)
                self.max_div = len(self.group)
            else:
                self.group.extend(nonseps)
            # initialize models
            self.initializeParameter()
            # initialize population
            self.pop = self.initializePopulation(self.pop)
            if self.cnf.abbrev_solution:
                self.pop.x = [ self.encodeCC(self.pop.x[_pop])  for _pop in range(len(self.pop.x)) ]
                self.pop.x_best = self.encodeCC(self.pop.x_best)
            # blank optimization
            if not self.blank_opt and blank_opt:
                self.pop = self.fillPositionwithBlank(self.pop)

        # evaluate x
        returns = self.getFitness(self.pop.x[_pop])
        # re-evaluation when error occurs
        if returns == 'No Fitness':
            for reeval in range(self.cnf.max_reeval):
                returns = self.getFitness(self.pop.x[_pop])
                if returns == 'No Fitness':
                    break
            else:
                print(f'[Optimizer] Error: Re-evaluate fitness for {self.cnf.max_reeval}-times, but return no fitness value in {self.fnc.prob_name} (class {self.__class__.__name__})', file=sys.stderr)
                raise Exception(f'Error: Re-evaluate fitness for {self.cnf.max_reeval}-times, but return no fitness value in {self.fnc.prob_name} (class {self.__class__.__name__})')
        else:
            # no fitness -> pass
            pass
        self.pop.f_new, self.pop.f_org_new, self.pop.NoVW_new, self.pop.NoVL_new, self.pop.totalWL_new, self.pop.calc_time = returns
        self.pop.f[_pop] = self.pop.f_new
        # update x_best, f_best
        best_index = self.getBestIndices(self.pop.f,1)
        self.pop = self.updateBest(self.pop, self.pop.x[best_index], self.pop.f[best_index])
        # record initial evaluations
        self.init_evals = self.fnc.total_evals
        # update indices
        self.updateIndices()
        # reset indices (when last update)
        if self.indices['div'] == 1:
            self.indices['div'], self.indices['pop'] = 0, 0


    def update(self, blank_opt:bool=True) -> None:
        '''
            update method (external reference)
            (one-call -> one-evaluation)
        '''
        if not self.blank_opt and blank_opt:
            self.pop = self.fillPositionwithBlank(self.pop)
        self.updateParameter()
        self.pop = self.updatePopulation(self.pop)
        self.updateIndices()


    def grouping(self) -> tuple[list,list]:
        '''
            grouping by any grouping method
        '''
        group_family, group_name = 'SG', 'layer'
        group_pdir = Stdio.makeDirectory(self.cnf.opt_path, self.cnf.dirname['group'], confirm=False)
        group_dir = Stdio.makeDirectory(group_pdir, group_family, confirm=False)
        group_file = os.path.join(group_dir, f'group_{group_name}_{self.fnc.prob_name}.csv')
        # Exist group file -> Import
        if os.path.isfile(group_file):
            print('[Optimizer] import group from {} ...'.format(group_file.split(os.path.sep)[-1]))
            # import seps and nonseps group from file
            df = Stdio.readDatabase(group_file).T
            # data organization
            group_evals = int(df.loc['total_evals',0])
            seps_with_nan = np.array(df[df.index.str.startswith('sep')].values)
            nonseps_with_nan = np.array(df[df.index.str.startswith('nonsep')].values)
            seps, nonseps = [], []
            for _sep in seps_with_nan:
                seps.append(list(_sep[~pd.isnull(_sep)])[0])
            for _nonsep in nonseps_with_nan:
                nonseps.append(list(_nonsep[~pd.isnull(_nonsep)]))
            self.fnc.total_evals += group_evals
        # Not exist group file -> Calculate and output
        else:
            print('[Optimizer] calculate group ...')
            # calculate seps and nonseps group
            _df = {}
            group_evals_s = self.fnc.total_evals
            seps = []
            nonseps = [ _dom for _dom in self.fnc.domain.values() ]
            group_evals_e = self.fnc.total_evals
            # data organization
            _df['evals'] = pd.DataFrame([group_evals_e - group_evals_s], index=['total_evals'])
            _df['seps'] = pd.DataFrame(seps, index=['sep_{}'.format(k) for k in range(len(seps))])
            _df['nonseps'] = pd.DataFrame(nonseps, index=['nonsep_{}'.format(k) for k in range(len(nonseps))])
            df = pd.concat([_df['evals'], _df['seps'], _df['nonseps']])
            df_column = pd.DataFrame([np.arange(len(df.columns),dtype=int)], index=['#index'], dtype=str)
            df = pd.concat([df_column, df]).T
            # output file
            if not os.path.isdir(group_dir):
                os.makedirs(group_dir)
            Stdio.writeDatabase(df, group_file)
        return seps, nonseps


    def __del__(self):
        del self.cnf, self.fnc



if __name__ == '__main__':
    import config           as cf
    import function         as fc
    import logger           as lg
    import dataprocessing   as dp
    import os
    # (1) working directory movement
    from logger import Stdio
    Stdio.moveWorkingDirectory()

    cnf = cf.Configuration()
    log_settings = lg.LogData(cnf)

    i,j = 0,0
    log = lg.LogData(cnf, cnf.prob_name[i])
    fnc = fc.Function(cnf, cnf.prob_name[i],1)
    opt = eval('{}(cnf, fnc)'.format(cnf.opt_name.replace('-','')))
    plt = dp.Plot(cnf, fnc)
    cnf.setRandomSeed(seed=j+1)
    # initialize optimizer
    opt.initialize(opt.blankOpt(fnc.total_evals))

