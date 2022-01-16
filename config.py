# Configuration
# version 1.5 (2022/01/15)

import  os
import  sys
import  time    as tm
import  numpy   as np
from    utils   import log
import shutil   as st
from joblib     import Parallel, delayed

class Configuration:
    '''Configuration
    '''

    def __init__(self):

        # Current time
        self.time = tm.strftime('%Y-%m-%d_%H-%M-%S')

        # Parallel Process
        self.n_jobs                 = -1                     # the number of CPU cores using for parallel processing
        '''
        1       : sequential execution          (cpu usage rate: ■□□□□□□□□□, ■=1)
        2       : 2 CPUs are used.              (cpu usage rate: ■■□□□□□□□□, ■=2)
        n (>0)  : n CPUs are used.              (cpu usage rate: ■■■■□□□□□□, ■=n)
        -1      : all CPUs are used.            (cpu usage rate: ■■■■■■■■■■, □=0)
        -2      : all CPUs but one are used.    (cpu usage rate: ■■■■■■■■■□, □=1)
        n (<0)  : all CPUs but -n-1 are used.   (cpu usage rate: ■■■■■■□□□□, □=-n-1)
        '''

        # Development Version
        self.version                = '1.0'

        # Problem (Function)
        prob_sets      = {
            # basic function
            'basic' : ['F1','F2','F3','F4','F5','F6','F7','F8','F9'],
            # LSGO2013 function
            'lsgo2013'  : ['LSGO2013_F1', 'LSGO2013_F2', 'LSGO2013_F3', 'LSGO2013_F4', 'LSGO2013_F5', 'LSGO2013_F6', 'LSGO2013_F7', 'LSGO2013_F8', 'LSGO2013_F9', 'LSGO2013_F10', 'LSGO2013_F11', 'LSGO2013_F12', 'LSGO2013_F13', 'LSGO2013_F14', 'LSGO2013_F15'],
            'lsgo2013-2div': [
                # 2div - 1st problem set
                ['LSGO2013_F2', 'LSGO2013_F3', 'LSGO2013_F4', 'LSGO2013_F5', 'LSGO2013_F6', 'LSGO2013_F7', 'LSGO2013_F8', 'LSGO2013_F9', 'LSGO2013_F10', 'LSGO2013_F11', 'LSGO2013_F12'],
                # 2div - 2nd problem set
                ['LSGO2013_F1', 'LSGO2013_F13', 'LSGO2013_F14', 'LSGO2013_F15']
            ]
        }
        self.prob_dim               = 1000
        self.prob_name              = prob_sets['lsgo2013']
        self.prob_env_noise         = 'off'                 # LSGO2013 benchmark noise (on/off)
        self.opt_type               = 'AutoComplete'        # AutoComplete / min / max

        # Environmental setting
        # self.max_trial              = 25                    # max trials (11/25/31)
        # self.max_evals              = 3_000_000             # max FEs (1_000_000/3_000_000/10_000_000)
        self.max_trial              = 2                     # max trials (11/25/31)
        self.max_evals              = 100_000               # max FEs (1_000_000/3_000_000/10_000_000)
        self.initial_seed           = 1                     # initial_seed ~ initial_seed + max_trial - 1

        # Optimizer setting
        ## General
        self.opt_name               = 'CCEA'                # optimizer name
        self.subopt_name            = 'PSO'                 # PSO, GA
        self.max_pop                = 10                    # population size (10)
        self.init_method            = 'random'              # random / lhs
        ## CC Parameter
        self.fitness_assignment     = 'best'                # best / current (*)
        self.group_name             = 'RDG3'                # SG / RG / RDG2 /RDG3
        self.deterministic_grouping = True                  # True / False
        self.nonseps_div            = False                 # nonseps division
        self.max_nonseps_dim        = 100                   # max nonseparable dimension
        ### SG/RG Parameter
        self.max_div                = 20                    # division number
        ### RDG2/RDG3 Parameter
        self.ep_n                   = 50                    # threshold min element size
        '''
        when ep_n = 1000, RDG3 is equivalent to the RDG or RDG2
        '''
        self.ep_s                   = 100                   # threshold max dimension value to solve separable sets
        ## GA Parameter
        self.crossover_rate         = 1.0
        self.mutation_rate          = 0.01
        ## PSO Parameter
        self.inertia                = 0.8
        self.accel_g                = 1.0
        self.accel_p                = 0.8

        # I/O setting
        ## folder/file name assignment
        ### frame directory
        self.dirname    = {
            # [1] root folder
            'env'           : '_env',
            'group'         : '_group',
            'result'        : '_result',
            # [2] _env folder
            'cec2013lsgo'   : 'cec2013lsgo',
            # [3] _result/exp_name
            'standard'      : '_std',
            'population'    : '_pop',
            'grouping'      : '_group',
            # [4] _result/exp_name/prob_name folder
            'trial'         : lambda n: f'trial-{n}',
            'trials'        : 'trials'
        }
        ### experiment directory
        self.comment    = f'{self.opt_name}-{self.subopt_name}_v{self.version}_{self.shortenEvals()}-evals_{self.time}'
        # self.comment    = f'debug-mode_{self.time}'
        self.expname    = {
            'result': f'result_{self.comment}'
        }
        ### experiment file
        self.filename   = {
            # _env
            'module'        : 'module.txt',
            # _group
            'group'         : lambda n: f'group_{n}.csv',
            # _result
            'setting'       : 'config.yml',
            # DataLogger
            'regular-log'   : lambda n,c: f'trial{n}_std({c}).xlsx',
            'result'        : lambda n: f'trial{n}_std.xlsx',
            'result-pop'    : lambda n: f'trial{n}_pop.csv',       # Recommend:csv (xlsx extension may break)
            'profile-report': lambda n: f'trial{n}_profile-report.html',
            # DataProcessing
            'result-all'    : lambda p,i: f'all_trials_{i}_{p}.xlsx',
            'result-stat'   : lambda p,i: f'stat-{i}_{p}.xlsx',
            'result-stat-image': lambda p,i: f'stat-{i}_{p}.png',
            'grouping'      : lambda g: f'{g}.png',
            'visual'        : lambda n,v,p: f'trial{n}_{v}_{p}.png',
            'fit-div'       : lambda n,p: f'trial{n}_bestfit-divcurve_{p}.xlsx',
            'fit-div-image' : lambda n,p: f'trial{n}_bestfit-divcurve_{p}.png'
        }

        ## output root path
        paths       = {
            # current directory from Terminal
            'current'   : os.getcwd(),
            # parent directory from this file
            'pardir'    : os.path.dirname(__file__)
        }
        ### path output
        self.path_out   = paths['pardir']
        ### frame path
        self.env_path = os.path.join(self.path_out, self.dirname['env'])
        self.result_path = os.path.join(self.path_out, self.dirname['result'])

        ## Logger
        self.log = {
            # standard log
            'standard'      : {
                'out'       :   True,       # output mode
                'n_div'     :   10,         # file division log
                'n_sample'  :   1000        # sample number
            },
            # population analysis log
            'population'    : {
                'out'       :   True,       # output mode
                'n_sample'  :   400,        # sample number
                'trial'     :   'all',      # object of trial ( first-only / all )
                'report'    :   False,      # output: profile report
                'visual'    :   False,      # output: tSNE
                'diversity' :   True        # output: diversity curve
            },
            # grouping image
            'grouping'      : {
                'out'       :   True        # ouput mode
            }
        }
        ## Data processing
        self.dlog_trials    = 'all'                             # output trial for detail log
        '''
            first-only    : output only first trial
            all           : output all trials
        '''

    def setRandomSeed(self, seed=1):
        '''Set seed value

        Args:
            seed (int, optional): Seed value. Defaults to 1.
        '''
        self.seed           = seed
        self.rd             = np.random                         # random
        self.rd.seed(self.seed)                                 # set seed for random

    def shortenEvals(self):
        '''Shorten fitness evaluations
        EX) 10,000 -> 10k  /  1,000,000 -> 1m

        Returns:
            str: shorten evaluation expression
        '''
        if self.max_evals//(10**3) > 0:
            if self.max_evals//(10**6) > 0:
                short_eval = f'{self.max_evals//(10**6)}m'
            else:
                short_eval = f'{self.max_evals//(10**3)}k'
        else:
            short_eval = f'{self.max_evals}'
        return short_eval

    def addComment(self, comment:str):
        '''Add comment

        Args:
            comment (str): comment for experiment log
        '''
        if not comment.startswith('_'):
            comment = f'_{comment}'
        self.comment += comment
        self.expname    = {
            'result': f'result_{self.comment}'
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

    def deleteFolders(self):
        # set absolute path
        root = os.getcwd()
        # delete folders path
        rem_paths = [os.path.join(root,self.dirname['result'],self.expname['result'])]
        # Parallel Process
        Parallel(n_jobs=self.n_jobs)( [delayed(self._deleteFolder)(rem_path)  for rem_path in rem_paths ] )
        print('[Config] Delete Root Folder.')


    @staticmethod
    def setup(filename:str):
        '''Setup PC

        Args:
            filename (str): module file path or filename
        '''
        import subprocess as sp
        PYTHON_VERSION:str  = '3.10'
        cmd = ('py', f'-{PYTHON_VERSION}', '-m', 'pip', 'install', '-r', filename)
        if os.path.isfile(filename):
            proc = sp.Popen(cmd, shell=True)
            result = proc.communicate()
            log('Configuration', f'Setup finish!')
        else:
            log('Configuration', f'Error: Do not exist path "{filename}"', output=sys.stderr)


''' main '''
if __name__ == '__main__':
    # setup PC for installing modules
    Configuration.setup('_env/module.txt')