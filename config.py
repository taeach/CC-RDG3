# Configuration
# version 2.6 (2021/12/1)

import os
import time  as tm          # standard library
import numpy as np
import shutil as st
from joblib import Parallel, delayed


''' Configuration '''
class Configuration:

    def __init__(self):

        # Get current time
        self.time = tm.strftime('%Y-%m-%d_%H-%M-%S')

        # Parallel Process
        self.n_jobs             = 9                     # the number of CPU cores using for parallel processing
        '''
            -1      : using all CPU cores(=n_cpus)
        '''

        """ Experimental Setting """
        # problem
        prob_sets      = [
            # 'test',
            '4-bitAdder',   '4-bitMultiplier',  '4-bitShiftRotator',
            '8-bitAdder',   '8-bitMultiplier',
            '16-bitAdder',  '16-bitMultiplier'
            ]

        prob_set = {
            # bit
            '4bit'  : ['4-bitAdder', '4-bitMultiplier', '4-bitShiftRotator'],
            '8bit'  : ['8-bitAdder', '8-bitMultiplier'],
            '16bit' : ['16-bitAdder', '16-bitMultiplier'],
            # equivalent problem computation
            'div2-1': ['4-bitAdder', '4-bitMultiplier', '4-bitShiftRotator', '16-bitAdder'],
            'div2-2': ['8-bitAdder', '8-bitMultiplier', '16-bitMultiplier'],
            # others
            '4-8bit'  : ['4-bitAdder', '4-bitMultiplier', '4-bitShiftRotator', '8-bitAdder', '8-bitMultiplier'],
            'part1-1': ['4-bitAdder', '4-bitMultiplier', '4-bitShiftRotator'],
            'part1-2': ['8-bitAdder', '8-bitMultiplier', '16-bitAdder', '16-bitMultiplier'],
            'part2': ['8-bitMultiplier', '16-bitAdder', '16-bitMultiplier']
        }

        self.prob_name          = prob_set['part2']
        self.opt_type           = 'AutoComplete'        # min or max
        # Environmental setting
        self.max_trial          = 5                     # max trials (11/25)
        self.max_evals          = 3_000_000             # max FEs (1_000_000/3_000_000/10_000_000)
        self.initial_seed       = 1                     # initial_seed ~ initial_seed + max_trial - 1

        # Optimizer setting
        ## General
        self.opt_name           = 'CCVR'                # optimizer type (the same class name except for '-')
        self.subopt_name        = 'ACO'                 # GA, ACO
        self.ls_name            = '2-opt'               # None, 2-opt, 3-opt
        # self.ls_name            = [None,'2-opt']        # None, 2-opt, 3-opt
        self.version            = '3.5'                 # development version of the system
        self.max_pop            = 10                    # population size (10)
        self.blank_opt          = '2-stage'             # on, off, 2-stage
        self.switch_timing      = 0.5                   # 2-stage parameter
        # self.switch_timing      = [0., 0.2, 0.4, 0.6, 0.8, 1.0]
        ## GA Parameter
        self.crossover_rate     = 1.0
        self.mutation_rate      = 0.01
        ## ACO Parameter
        self.evaporation_rate   = 0.05
        self.confidential_rate  = 0.95
        self.beta               = 1
        ## CC Parameter
        self.abbrev_solution    = False
        self.fitness_assignment = 'best'                # best / current (*)
        ## Grouping Parameter
        self.nonseps_div        = False                 # nonseps division
        self.max_groupsize      = 100                   # max group size
        ## Evaluator
        self.max_reeval         = 10                    # re-eval
        self.function_type      = 'adaptive'            # lasso (L1) or ridge (L2) or original or adaptive

        # I/O setting
        ## folder/file name assignment
        ### frame directory
        self.dirname    = {
            # [1] root folder
            'env'   : '_env_easy',
            'log'   : '_log',
            'result': '_result',
            'eval'  : 'evaluator',
            'opt'   : 'optimizer',
            # [2] _log folder
            'input' : '_input',
            'job'   : '_job',
            'output': '_output',
            # [3] _output/prob_name folder
            'trial' : lambda n: f'trial-{n}',
            # [4] _result/prob_name folder
            'trials': 'trials',
            # [5] optimizer folder
            'group' : '_group'
        }
        ### experiment directory
        self.comment    = f'AQFPLogicOpt_{self.shortenEvals()}-evals_{self.opt_name}-{self.subopt_name}_{self.time}'
        # self.comment    = f'debug-mode_{self.time}'
        self.expname    = {
            'input' : f'input_ver{self.version}_{self.comment}',
            'job'   : f'job_ver{self.version}_{self.comment}',
            'output': f'output_ver{self.version}_{self.comment}',
            'result': f'result_ver{self.version}_{self.comment}'
        }
        ### experiment file
        self.filename   = {
            # _log
            'input' : lambda n: f'in_gen{n}.csv',
            'job'   : lambda n: f'job_gen{n}.csv',
            'output': lambda n: f'out_gen{n}.csv',
            'data'  : lambda n: f'dat_gen{n}.csv',
            # _result
            'result': lambda n: f'trial{n}_std.xlsx',
            'regular-log': lambda n,FEs: f'trial{n}_std_agg{FEs}.xlsx',
            'result-last': lambda n: f'trial{n}_std_last.xlsx',
            'regular-result': lambda n,FEs: f'trial{n}_std_best_agg{FEs}.xlsx',
            'result-all': lambda p,i: f'all_trials_{i}_{p}.xlsx',
            'result-stat': lambda p,i: f'stat-{i}_{p}.xlsx',
            'result-stat-image': lambda p,i: f'stat-{i}_{p}.png',
            'setting': 'config.yml',
            # evaluator
            # 'eval'  : lambda p='_',q='_',n='_': f'AQFPQoRChecker-gnu_{p}_{q}_{n}.exe'
            'eval'  : lambda p='_',q='_',n='_': f'AQFPQoRChecker-msvc_{p}_{q}_{n}.exe'
        }

        ## path
        ### output root path
        paths            = {
            'current'   : os.getcwd(),
            'Home-OD'   : r'E:\OneDrive - 横浜国立大学\Documents\Development\Evolutionary Computation\AQFPLogic-Optimization',
            'Laptop'    : r'C:\Users\taichi\Documents\SourceCode\AQFPLogic-Optimization',
            'Laptop-OD' : r'C:\Users\Yoshikawa\OneDrive - 横浜国立大学\Documents\Development\Evolutionary Computation\AQFPLogic-Optimization',
            'Lab-OD'    : r'C:\Users\Yoshikawa\OneDrive - 横浜国立大学\Documents\Development\Evolutionary Computation\AQFPLogic-Optimization',
            'CLUSTER'   : r'C:\Users\nktlab\Desktop\Experiment\Yoshikawa\AQFPLogicOpt',
            'RTX-01'    : r'D:\Experiment\Yoshikawa\AQFPLogicOpt'
        }
        # path output
        # current or CLUSTER or RTX-01
        # self.path_out   = paths['RTX-01']
        self.path_out   = paths['current']
        # nas settings
        self.nas_mode   = False
        # current
        self.path_nas   = paths['current']
        path_nas = self.path_nas if self.nas_mode else self.path_out
        ### frame path
        self.env_path = os.path.join(path_nas, self.dirname['env'])
        self.log_path = os.path.join(self.path_out, self.dirname['log'])
        self.result_path = os.path.join(path_nas, self.dirname['result'])
        self.opt_path = os.path.join(path_nas, self.dirname['opt'])
        self.eval_path = os.path.join(self.path_out, self.dirname['eval'])

        ## Logger
        self.log = {
            # standard log
            'standard'      : {
                'out'       :   True,
                'n_sample'  :   1000
            },
            # population analysis log
            'population'    : {
                'out'       :   False,
                'n_sample'  :   400,
                'trial'     :   'all'           # first-only or all
            }
        }
        ## Data processing
        self.dlog_trials    = 'all'                             # output trial for detail log
        '''
            first-only    : output only first trial
            all           : output all trials
        '''

        ## Animation
        self.anime_out      = False                             # output animation
        self.anime_interval = 5                                 # interval of animation output

    def setRandomSeed(self, seed=1):
        self.seed           = seed
        self.rd             = np.random                         # random
        self.rd.seed(self.seed)                                 # set seed for random

    def shortenEvals(self):
        if self.max_evals//(10**3) > 0:
            if self.max_evals//(10**6) > 0:
                short_eval = f'{self.max_evals//(10**6)}m'
            else:
                short_eval = f'{self.max_evals//(10**3)}k'
        else:
            short_eval = f'{self.max_evals}'
        return short_eval

    def addComment(self, comment:str):
        if not comment.startswith('_'):
            comment = f'_{comment}'
        self.comment += comment
        self.expname    = {
            'input' : f'input_ver{self.version}_{self.comment}',
            'job'   : f'job_ver{self.version}_{self.comment}',
            'output': f'output_ver{self.version}_{self.comment}',
            'result': f'result_ver{self.version}_{self.comment}'
        }

    @staticmethod
    def _deleteFolder(rem_path:str):
        try:
            if os.path.isdir(rem_path):
                st.rmtree(rem_path)
                print(f'[Config] Delete folder "{rem_path}".')
            else:
                print(f'[Config] Warning: Folder "{rem_path}" not exist.')
        except Exception as e:
            print(f'[Config] Error: {e}')

    def deleteFolders(self, mode:str='log'):
        # set absolute path
        if mode=='log':
            log_path = self.log_path
        elif mode=='all':
            root = os.getcwd()
            log_path = os.path.join(root,self.dirname['log'])
        else:
            raise Exception(f'Error: invalid mode "{mode}" in deleteFolders')
        # delete folders path
        rem_paths = [
            os.path.join(log_path,self.dirname['input'],self.expname['input']),
            os.path.join(log_path,self.dirname['job'],self.expname['job']),
            os.path.join(log_path,self.dirname['output'],self.expname['output'])
        ]
        if mode=='all':
            rem_paths.append(
                os.path.join(root,self.dirname['result'],self.expname['result'])
            )
        # Parallel Process
        Parallel(n_jobs=self.n_jobs)( [delayed(self._deleteFolder)(rem_path)  for rem_path in rem_paths ] )
        print('[Config] Delete Process Finished!')


''' main '''
if __name__ == '__main__':
    print('< Folder deletion >')
    # (1) working directory movement
    from logger import Stdio
    Stdio.moveWorkingDirectory()
    # (2) delete folders
    cnf = Configuration()
    while(True):
        print('■ Please input filename below:')
        print('[EX1] input_verX.X_probname_optimizer_2021-06-14_02-30-15')
        print('[EX2] job_verX.X_probname_optimizer_2021-06-14_02-30-15')
        print('[EX3] output_verX.X_probname_optimizer_2021-06-14_02-30-15')
        print('[EX4] result_verX.X_probname_optimizer_2021-06-14_02-30-15')
        filename = input('filename : ')
        if filename == '':
            break
        _discard, cnf.version, cnf.comment = filename.split('_',2)
        cnf.version = cnf.version.replace('ver','')
        cnf.expname    = {
            'input' : f'input_ver{cnf.version}_{cnf.comment}',
            'job'   : f'job_ver{cnf.version}_{cnf.comment}',
            'output': f'output_ver{cnf.version}_{cnf.comment}',
            'result': f'result_ver{cnf.version}_{cnf.comment}'
        }
        cnf.deleteFolders('all')