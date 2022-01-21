# Optimizer
# version 1.7 (2022/01/15)

import os
import time             as tm
from typing             import Callable
import math             as mt
import numpy            as np
import pandas           as pd
from suboptimizer       import Population, PSO
from utils              import Stdio, log

''' For Function Annotations '''
from config             import Configuration
from function           import Function
from logger             import DataLogger


class CCEA(eval(Configuration().subopt_name)):
# For VSCode Reference
# class CCEA(PSO):
    '''CCEA (Cooperative Co-Evolutionary Algorithm)

    Attributes:
        cnf : Configuration class
        fnc : Function class
        dlg : DataLogger class

    HowToUse:
    - Initialize(blank_opt)
    - update(blank_opt) (repeat for termination)

    Note:
        - One call, One evaluation framework
    '''
    def __init__(self, cnf:Configuration, fnc:Function, dlg:DataLogger) -> None:
        super().__init__(cnf, fnc)
        self.dlg = dlg
        self.pop = Population()
        self.max_div = None
        self.init_evals = 0


    def initialize(self) -> None:
        '''initialize method (external reference)
           (one-call -> one-evaluation)
        '''
        _div, _pop = self.getIndices
        if _pop == 0:
            # set opt type
            self.setOptType(self.fnc.prob_name)
            # grouping
            if self.cnf.group_name == 'SG':
                grouping_fnc = self.SG
            elif self.cnf.group_name == 'RG':
                grouping_fnc = self.RG
            elif self.cnf.group_name in ('RDG2','RDG3'):
                grouping_fnc = self.RDG3
            seps, nonseps = self.grouping(grouping_fnc)
            self.group = seps.copy() if seps != [] else []
            if self.cnf.nonseps_div:
                # divide max_nonseps_dim
                for _nonsep in nonseps:
                    if len(_nonsep) > self.cnf.max_nonseps_dim:
                        loop = mt.ceil(len(_nonsep)/self.cnf.max_nonseps_dim)
                        for _loop in range(loop):
                            self.group.append(_nonsep[_loop*self.cnf.max_nonseps_dim:(_loop+1)*self.cnf.max_nonseps_dim-1])
                    else:
                        self.group.append(_nonsep)
            else:
                self.group.extend(nonseps)
            self.max_div = len(self.group)
            self.dim = [ len(group) for group in self.group ]
            # initialize models
            self.initializeParameter()
            # initialize population
            self.pop = self.initializePopulation(self.pop)

        # evaluate x
        self.pop.x_new = self.linkSolution(self.pop)
        self.pop.f_new = self.getFitness(self.pop.x_new)
        self.init_evals = self.fnc.total_evals
        while _pop == self.getIndices[1]:
            # update x_best, f_best
            self.pop = self.updateBest(self.pop, self.pop.x_new, self.pop.f_new)
            # update indices
            self.updateIndices('div')


    def update(self) -> None:
        '''update method (external reference)
           (one-call -> one-evaluation)
        '''
        self.updateParameter()
        self.pop = self.updatePopulation(self.pop)
        self.updateIndices('pop')


    def grouping(self, grouping_fnc:Callable) -> tuple[np.ndarray, np.ndarray]:
        '''grouping by any grouping method

        Note:
            group_path(str): path of grouping file to use
        '''
        assert grouping_fnc.__name__ in self.static_grouping + self.random_grouping + self.dynamic_grouping, f'Error: Please add grouping_name "{grouping_fnc.__name__}" to grouping ({self.__class__.__name__}.grouping)'
        if grouping_fnc.__name__ in self.static_grouping:
            group_family = 'SG'
        elif grouping_fnc.__name__ in self.random_grouping:
            group_family = 'RG'
        elif grouping_fnc.__name__ in self.dynamic_grouping:
            group_family = 'DG'
        group_pdir = Stdio.makeDirectory(self.cnf.path_out, self.cnf.dirname['group'], confirm=False)
        group_dir = Stdio.makeDirectory(group_pdir, group_family, confirm=False)
        # Generate filename
        if grouping_fnc.__name__ in self.static_grouping + self.random_grouping:
            self.group_path = os.path.join(group_dir, self.cnf.filename['group'](f'{self.cnf.group_name}_{self.fnc.prob_dim}D_{self.cnf.max_div}div'))
        elif grouping_fnc.__name__ in self.dynamic_grouping:
            self.group_path = os.path.join(group_dir, self.cnf.filename['group'](f'{self.cnf.group_name}_{self.fnc.prob_name}_{self.cnf.prob_env_noise.upper()}'))
        if not self.cnf.deterministic_grouping and not grouping_fnc.__name__ in self.static_grouping:
            path, ext = os.path.splitext(self.group_path)
            self.group_path = f'{path}_seed={self.cnf.seed}{ext}'

        self.f_log = []
        # Not exist group file -> Calculate and output group file
        if not os.path.isfile(self.group_path):
            log(self, f'Calculate group by {self.cnf.group_name} ({os.path.basename(self.group_path)}) ...')
            # calculate seps and nonseps group
            _df = {}
            evals_s, time_s = self.fnc.total_evals, tm.mktime(tm.localtime())
            seps, nonseps = grouping_fnc()
            evals_e, time_e = self.fnc.total_evals, tm.mktime(tm.localtime())
            self.f_log = np.array(self.f_log)
            # data organization
            _df['exe_time'] = pd.DataFrame([time_e - time_s], index=['exe_time'])
            _df['evals'] = pd.DataFrame([evals_e - evals_s], index=['total_evals'])
            _df['fitness'] = pd.DataFrame([self.f_log], index=['fitness'])
            _df['seps'] = pd.DataFrame(seps, index=[f'sep_{k}' for k in range(len(seps))])
            _df['nonseps'] = pd.DataFrame(nonseps, index=[f'nonsep_{k}' for k in range(len(nonseps))])
            df = pd.concat([_df[key] for key in _df.keys()])
            df_column = pd.DataFrame([np.arange(len(df.columns),dtype=int)], index=['#index'], dtype=str)
            df = pd.concat([df_column, df]).T
            # output file
            if not os.path.isdir(group_dir):
                os.makedirs(group_dir)
            if not os.path.isfile(self.group_path):
                Stdio.writeDatabase(df, self.group_path)
        # Exist group file -> Import
        else:
            log(self, f'Import group from {os.path.basename(self.group_path)} ...')
            # import seps and nonseps group from file
            df = Stdio.readDatabase(self.group_path).T
            # data organization
            group_times = int(df.loc['exe_time',0])
            group_evals = int(df.loc['total_evals',0])
            self.f_log = np.array(df.loc['fitness'].values)
            seps_with_nan = np.array(df[df.index.str.startswith('sep')].values)
            nonseps_with_nan = np.array(df[df.index.str.startswith('nonsep')].values)
            seps, nonseps = [], []
            for _sep in seps_with_nan:
                seps.append(np.array(_sep[~pd.isnull(_sep)], dtype=int))
            for _nonsep in nonseps_with_nan:
                nonseps.append(np.array(_nonsep[~pd.isnull(_nonsep)], dtype=int))
            self.fnc.total_evals += group_evals
            self.dlg.addExeTime(group_times)

        return seps, nonseps


    def __del__(self):
        del self.cnf, self.fnc



if __name__ == '__main__':
    # test
    from config     import Configuration
    from function   import Function
    from logger     import DataLogger
    from processing import DataProcessing
    import os
    # (1) working directory movement
    from utils import Stdio
    Stdio.moveWorkingDirectory()
    cnf = Configuration()
    # settings
    cnf.max_evals = 100
    cnf.prob_name = 'LSGO2013_F1'
    seed = 1
    dlg = DataLogger(cnf, cnf.prob_name)
    fnc = Function(cnf, cnf.prob_name)
    opt = eval(f'{cnf.opt_name}(cnf, fnc, dlg)')
    cnf.setRandomSeed(seed)

    for k in range(cnf.max_pop):
        opt.initialize()
    cnf.max_evals += opt.init_evals
    while fnc.total_evals < cnf.max_evals :
        opt.update()
