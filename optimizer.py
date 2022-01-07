# Optimizer
# version 1.0 (2021/12/17)

import os
import sys
import math             as mt
import numpy            as np
import pandas           as pd
from suboptimizer       import Population, GA, PSO
from utils              import Stdio, log

''' For Function Annotations '''
from config             import Configuration
from function           import Function


# class CCEA(eval(Configuration().subopt_name)):
class CCEA(GA):
    '''CCEA (Cooperative Co-Evolutionary Algorithm)

    Attributes:
        cnf : Configuration class
        fnc : Function class

    HowToUse:
    - Initialize(blank_opt)
    - update(blank_opt) (repeat for termination)
    '''
    def __init__(self, cnf:Configuration, fnc:Function) -> None:
        super().__init__(cnf, fnc)
        self.pop = Population()
        # static grouping
        self.max_div = self.cnf.max_div
        self.init_evals = 0


    def initialize(self) -> None:
        '''initialize method (external reference)
           (one-call -> one-evaluation)
        '''
        _pop = self.indices['pop']
        if _pop == 0:
            # set opt type
            self.setOptType(self.fnc.prob_name)
            # grouping
            if self.cnf.group_name == 'static':
                pass
            elif self.cnf.group_name in ('RDG2','RDG3'):
                seps, nonseps = self.grouping(self.RDG3)
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

        # evaluate x
        if isinstance(f_new:=self.getFitness(self.pop.x[_pop]),float):
            self.pop.f_new = f_new
            self.pop.f[_pop] = f_new
        else:
            log('Optimizer', f'Error: {f_new}', output=sys.stderr)
            raise Exception(f'Error: {f_new}')
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


    def update(self) -> None:
        '''update method (external reference)
           (one-call -> one-evaluation)
        '''
        self.updateParameter()
        self.pop = self.updatePopulation(self.pop)
        self.updateIndices()


    def grouping(self, grouping_fnc:function) -> tuple[list]:
        '''grouping by any grouping method
        '''
        if grouping_fnc in ('RDG3'):
            group_family = 'DG'
        elif grouping_fnc in ('SG'):
            group_family = 'SG'
        group_pdir = Stdio.makeDirectory(self.cnf.path_out, self.cnf.dirname['group'], confirm=False)
        group_dir = Stdio.makeDirectory(group_pdir, group_family, confirm=False)
        group_file = os.path.join(group_dir, f'group_{self.cnf.group_name}_{self.fnc.prob_name}.csv')
        # Exist group file -> Import
        if os.path.isfile(group_file):
            log('Optimizer', f'Import group from {os.path.basename(group_file)} ...')
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
            log('Optimizer', f'Calculate group by {self.cnf.group_name} ...')
            # calculate seps and nonseps group
            _df = {}
            group_evals_s = self.fnc.total_evals
            seps, nonseps = grouping_fnc()
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
    from config     import Configuration
    from function   import Function
    from logger     import DataLogger
    from processing import DataProcessing
    import os
    # (1) working directory movement
    from utils import Stdio
    Stdio.moveWorkingDirectory()

    cnf = Configuration()
    log_settings = DataLogger(cnf)

    i,j = 0,0
    log = DataLogger(cnf, cnf.prob_name[i])
    fnc = Function(cnf, cnf.prob_name[i],1)
    opt = eval('{}(cnf, fnc)'.format(cnf.opt_name.replace('-','')))
    cnf.setRandomSeed(seed=j+1)
    # initialize optimizer
    opt.initialize()

