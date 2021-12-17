# Data Logger
# version 18.4 (2021/12/1)

import os                       # standard library
import re                       # standard library
import glob         as gl       # standard library
import sys                      # standard library
import csv                      # standard library
import time         as tm       # standard library
import traceback    as tb       # standard library
import platform     as pf       # standard library
import subprocess   as sp       # standard library
import shutil       as st       # standard library
import types        as tp       # standard library
import numpy        as np
import pandas       as pd
import yaml
from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib import colors as cls
from tqdm import tqdm
from joblib import Parallel, delayed
''' For Function Annotations '''
import config           as cf

class LogData :
    def __init__(self, cnf:cf.Configuration, prob_name:str=None):

        self.cnf                = cnf               # Configuration class
        self.prob_name          = prob_name         # problem to solve
        self.prob_dim           = None              # problem dimension
        self.config             = None              # output config.yml
        self.std_db             = []                # standard database
        self.pop_db             = []                # population database
        self.oth_db             = []                # others database
        self.model_gen          = 1                 # model generation
        self.timer              = None              # timer
        self.exe_time           = 0                 # execution time
        self.total_exe_time     = 0                 # execution time (All trial, problem)
        self.average_exe_time   = 0                 # execution time (Average)

        self.path_out_settings  = os.path.join(self.cnf.result_path, self.cnf.expname['result'], self.cnf.filename['setting'])

        self.file_encoding      = 'UTF-8'
        ## frequency of the log output
        self.slog_evals  = [ (self.cnf.max_evals * _num)//self.cnf.log['standard']['n_sample']  for _num in range(1, self.cnf.log['standard']['n_sample']+1) ]
        self.regular_evals  = [ (self.cnf.max_evals * _num)//10  for _num in range(1, 10+1) ]
        self.dlog_evals  = [ (self.cnf.max_evals * _num)//self.cnf.log['population']['n_sample']  for _num in range(1, self.cnf.log['population']['n_sample']+1) ]
        self.added_evals = []
        # stdout digit
        self.trial_digit = len(str(self.cnf.max_trial))
        self.evals_digit = len(str(self.cnf.max_evals))
        self.fitness_digit = 10

        # set path
        if not prob_name is None:
            # for result log
            self.path_out = os.path.join(self.cnf.result_path, self.cnf.expname['result'], self.prob_name)
            self.path_trial = Stdio.makeDirectory(self.path_out, self.cnf.dirname['trials'])
        else:
            # for settings_file
            self.path_trial = Stdio.makeDirectory(self.cnf.result_path, self.cnf.expname['result'])

        # make directory
        if self.cnf.log['standard']:
            if not os.path.isdir(self.path_trial):
                os.makedirs(self.path_trial)


    def startStopwatch(self):
        '''Start Stopwatch for exe-time counter
        '''
        if self.timer is None:
            self.timer = tm.mktime(tm.localtime())
        else:
            print(f'[LogData] Error: Try again after stopping the stopwatch. [startStopwatch -> startStopwatch] (class {self.__class__.__name__})', file=sys.stderr)


    def stopStopwatch(self):
        '''Stop Stopwatch for exe-time counter
        '''
        if not self.timer is None:
            self.exe_time += int(tm.mktime(tm.localtime()) - self.timer)
            self.timer = None
        else:
            print(f'[LogData] Error: Try again after starting the stopwatch. [ ? -> stopStopwatch] (class {self.__class__.__name__})', file=sys.stderr)


    def outSetting(self, timing:str, stdout:bool=True, timer:bool=True):
        '''output Setting File

        Args:
            timing (str): Output timing (`start` or `end`)
            stdout (bool, optional): Standard output. Defaults to True.
            timer (bool, optional): start or stop timer. Default to True.

        Raises:
            ValueError: Invalid timing.
        '''
        if self.cnf.log['standard']:
            if timing == 'start' :
                if timer:
                    self.startStopwatch()
                # get spec by wmic
                cpu_name = str(sp.check_output(['wmic','cpu','get', 'name'])).split('\\n')[1].replace('\\r','').strip()
                cpu_phy_core = str(sp.check_output(['wmic','cpu','get', 'NumberOfCores'])).split('\\n')[1].replace('\\r','').replace(' ','')
                cpu_log_core = str(sp.check_output(['wmic','cpu','get', 'NumberOfLogicalProcessors'])).split('\\n')[1].replace('\\r','').replace(' ','')
                ram = round(int(str(sp.check_output(['wmic','computersystem','get', 'totalphysicalmemory'])).split('\\n')[1].replace('\\r',''))/1024/1024/1024, 1)
                os_name = str(sp.check_output(['wmic','os','get', 'caption'])).split('\\n')[1].replace('\\r','').strip()
                # config variables
                cnf_vars = deepcopy(vars(self.cnf))
                for key,val in list(cnf_vars.items()):
                    if isinstance(val, list):
                        for i in reversed(range(len(val))):
                            if isinstance(val[i],(tp.ModuleType,tp.FunctionType)):
                                del cnf_vars[key][i]
                    elif isinstance(val,dict):
                        for _key in list(val):
                            if isinstance(val[_key],(tp.ModuleType,tp.FunctionType)):
                                del cnf_vars[key][_key]
                    elif isinstance(val,(tp.ModuleType,tp.FunctionType)):
                        del cnf_vars[key]
                # output file
                self.config = {
                    'LogTime': {
                        'StartTime' : tm.strftime('%Y %b %d (%a)  %H:%M:%S'),
                    },
                    'Spec': {
                        'ComputerName'      : pf.node(),
                        'CPU':{
                            'Name'          : cpu_name,
                            'PhysicalCore'  : f'{cpu_phy_core} core',
                            'LogicalCore'   : f'{cpu_log_core} thread'
                        },
                        'Memory(RAM)'       : f'{ram} GB',
                        'OS':{
                            'Name'          : os_name,
                            'Version'       : pf.version()
                        },
                        'PythonVersion'     : pf.python_version()
                    },
                    'Settings': cnf_vars
                }
                # output config.yml
                config_yml = yaml.safe_dump(self.config, default_flow_style=False,sort_keys=False)
                with open(self.path_out_settings, 'x') as f:
                    f.write(config_yml)
                # standard output
                if stdout:
                    print(f'[LogData]\n{config_yml}')

            elif timing == 'end':
                if timer:
                    self.stopStopwatch()
                # add EndTime and ExeTime
                dif_time    = self.total_exe_time
                dif_time_dict = {'day':0,'hour':0,'minute':0,'second':0}
                dif_time_dict['second'] = dif_time % 60
                dif_time //= 60
                dif_time_dict['minute'] = dif_time % 60
                dif_time //= 60
                dif_time_dict['hour']   = dif_time % 24
                dif_time //= 24
                dif_time_dict['day']    = dif_time
                self.config['LogTime']['EndTime'] = tm.strftime('%Y %b %d (%a)  %H:%M:%S')
                self.config['LogTime']['TotalExeTime'] = {
                    'Seconds'   : self.total_exe_time,
                    'Format'    : f"{dif_time_dict['day']} day - {dif_time_dict['hour']} h - {dif_time_dict['minute']} min - {dif_time_dict['second']} sec"
                }
                dif_time    = self.average_exe_time
                dif_time_dict = {'day':0,'hour':0,'minute':0,'second':0}
                dif_time_dict['second'] = dif_time % 60
                dif_time //= 60
                dif_time_dict['minute'] = dif_time % 60
                dif_time //= 60
                dif_time_dict['hour']   = dif_time % 24
                dif_time //= 24
                dif_time_dict['day']    = dif_time
                self.config['LogTime']['EndTime'] = tm.strftime('%Y %b %d (%a)  %H:%M:%S')
                self.config['LogTime']['AverageExeTime'] = {
                    'Seconds'   : self.average_exe_time,
                    'Format'    : f"{dif_time_dict['day']} day - {dif_time_dict['hour']} h - {dif_time_dict['minute']} min - {dif_time_dict['second']} sec"
                }
                # output config.yml
                with open(self.path_out_settings, 'wt') as f:
                    f.write(yaml.safe_dump(self.config, default_flow_style=False,sort_keys=False))
                # standard output
                if stdout:
                    print(f'[LogData] * Execution time : {self.total_exe_time}[sec]')
            else :
                raise ValueError(f'Invalid argument timing={timing} in LogData.outSetting()')



    def logging(self, opt:cf.Configuration, total_evals:int, trial:int) :
        '''
            Organization function to summarize logging function
        '''
        # get problem dimension
        self.prob_dim = len(opt.pop.getBest[0])
        self.loggingStandard(opt, total_evals, trial, option='no-sort')
        self.loggingDetail(opt, total_evals, trial)


    def outLog(self, opt:cf.Configuration, total_evals:int, trial:int) :
        '''
            Organization function to summarize log-output function
        '''
        self.outLogStandard(opt, total_evals, trial, option='no-sort')
        self.outLogDetail(opt, total_evals, trial)


    def loggingStandard(self, opt:cf.Configuration, total_evals:int, trial:int, option:str='no-sort'):
        '''
            [sequential processing]
            Get standard log for Population class
        '''
        if self.outJudgement('loggingStandard', trial, total_evals) :
            if option == 'no-sort' :
                x_best, f_best, f_org_best, NoVW_best, NoVL_best, totalWL_best = opt.pop.getBest
                ls_count = opt.params['ls_count']
                f_new, f_org, NoVW_new, NoVL_new, totalWL_new, calc_time_new = opt.pop.getCurrent
            else:
                print(f'[LogData] Error : Do not exist loggingStandard option {option} (class {self.__class__.__name__})')
                return

            # standard log
            # [ evals, CalcTime, Fitness, OriginalFitness, TotalWireLength, NumberOfViolationWires, NumberOfViolationLayer, FitnessValue, OriginalFitnessValue, TotalWireLength, NoVW, NoVL]
            dtset = [total_evals, calc_time_new, f_new, f_org, totalWL_new, NoVW_new, NoVL_new, f_best, f_org_best, totalWL_best, NoVW_best, NoVL_best, ls_count]
            if not total_evals in self.added_evals:
                self.std_db.append(dtset)
                self.added_evals.append(total_evals)

            # aggregate standard log
            if total_evals in self.regular_evals:
                count = self.regular_evals.index(total_evals)
                start_FEs = self.regular_evals[count-1] if count!=0 else 0
                end_FEs = self.regular_evals[count]
                assert end_FEs==total_evals, 'Warning: Unexpected value in aggregate standard log'
                # create header
                std_head = ['FEs', 'CalcTime[ms]', 'CurrentFV', 'CurrentOrgFV','CurrentTotalWL', 'CurrentNoVW', 'CurrentNoVL', 'FitnessValue', 'OrgFitnessValue', 'TotalWL', 'NoVW', 'NoVL', 'LSCount']
                # create dataframe
                std_df = pd.DataFrame(self.std_db, columns=std_head)
                dif_std_df = std_df.query(f'{start_FEs} < FEs <= {end_FEs}')
                # save log
                path_log_dif = os.path.join(self.path_trial, self.cnf.filename['regular-log'](trial,count+1))
                Stdio.writeDatabase(dif_std_df, path_log_dif)

            # last standard log
            # if totalevals == self.slog_evals[-1]:
            if total_evals in self.regular_evals:
                count = self.regular_evals.index(total_evals)
                info_best = {
                    'FEs'           : [total_evals],
                    'FitnessValue' : [f_best],
                    'OriginalFitnessValue' : [f_org_best],
                    'NumberOfViolationWires' : [NoVW_best],
                    'NumberOfViolationLayer' : [NoVL_best]
                }
                for layer,x_layer in enumerate(x_best):
                    info_best[f'x (layer{layer+1})'] = x_layer
                # param
                info_best['-'*10] = ['-'*10]
                for key,val in opt.params.items():
                    info_best[key] = [val]
                Stdio.saveExperimentalData(self.path_trial, info_best, self.cnf.filename['regular-result'](trial,count+1), display='horizontal')
                self.added_evals = []

            self.added_evals.append(total_evals)


    def loggingDetail(self, opt:cf.Configuration, total_evals:int, trial:int):
        '''
            [sequential processing]
            Get detail log for Population class
        '''
        if self.outJudgement('loggingDetail', trial, total_evals) :
            if self.cnf.log['population']['out']:
                x, f = opt.pop.getCurrent
                # population db
                # [ #evals, j, f_j, x_j[0],..., x_j[dim], mu_j[0],...,mu_j[dim] ]
                for k in range(self.cnf.max_pop) :
                    _pop_dt = [ total_evals, k, f[k]]
                    _pop_dt.extend(x[k])
                    self.pop_db.append(_pop_dt)

            if self.cnf.log['population']['out']:
                x, f = opt.pop.getCurrent
                # others db
                # Calculate Diversity Measure
                x_mean = np.sum(x , axis=0) / self.cnf.max_pop
                div_x = np.sum(np.sum((x - x_mean)**2, axis=1)**(1/2))
                # [ eval, div_x, model_gen, ]
                _oth_db = [ total_evals, div_x, opt.surrogate.gen, opt.surrogate.score, opt.surrogate.feature_importance, opt.surrogate.regen_cnt, opt.surrogate.f_newpred]
                self.oth_db.append(_oth_db)


    def outLogStandard(self, opt:cf.Configuration, total_evals:int, trial:int, option:str='no-sort'):
        '''
            [sequential processing]
            standard log-output for Population class
        '''
        # csv output
        if self.outJudgement('outLogStandard', trial) :
            # create header
            std_head = ['FEs', 'CalcTime[ms]', 'CurrentFV', 'CurrentOrgFV','CurrentTotalWL', 'CurrentNoVW', 'CurrentNoVL', 'FitnessValue', 'OrgFitnessValue', 'TotalWL', 'NoVW', 'NoVL', 'LSCount']
            # create dataframe
            std_df = pd.DataFrame(self.std_db, columns=std_head)
            # save log
            path_std = os.path.join(self.path_trial, self.cnf.filename['result'](trial))
            Stdio.writeDatabase(std_df, path_std)

        # standard output
        if option == 'no-sort' :
            x_best, f_best, f_org_best, NoVW_best, NoVL_best, totalWL_best = opt.pop.getBest
        else:
            print(f'[LogData] Error : Do not exist loggingStandard option {option} (class {self.__class__.__name__})')
            return

        print('[{}: trial-{} BEST ]  FEs = {}  |  f(x) = {}  |  TotalWL = {}  |  NoVW = {}  |  NoVL = {}'.format(
                self.prob_name.ljust(16),
                str(trial).ljust(self.trial_digit),
                str(total_evals).rjust(self.evals_digit),
                str(f_best)[:self.fitness_digit].ljust(self.fitness_digit),
                str(int(totalWL_best)).ljust(8),
                str(int(NoVW_best)).ljust(4),
                str(int(NoVL_best)).ljust(2)))
        # reset data
        self.std_db= []


    def outLogDetail(self, opt:cf.Configuration, total_evals:int, trial:int):
        '''
            [sequential processing]
            detail log-output for Population class
        '''
        # csv output
        if self.outJudgement('outLogDetail', trial) :
            if self.cnf.log['population']['out']:
                # create header
                # [ #evals, k, fk(x), x_k[0],..., x_k[n] ]
                pop_head = ['FEs', 'k', 'fk(x)']
                pop_head.extend(['xk[{}]'.format(i) for i in range(self.prob_dim)])
                # pop_head.extend(['mu_{}'.format(i) for i in range(self.prob_dim)])
                # save data frame
                path_pop = self.path_trial +'/trial{}_pop.csv'.format(trial)
                pop_df = pd.DataFrame(self.pop_db, columns=pop_head)
                Stdio.writeDatabase(pop_df, path_pop)

            if self.cnf.log['population']['out']:
                path_oth = self.path_trial +'/trial{}_others.csv'.format(trial)
                oth_head = ['FEs', 'D(x)', 'SM:gen', 'SM:score', 'SM:feature_importance', 'SM:regen_cnt', 'SM:f_pred']
                oth_df = pd.DataFrame(self.oth_db, columns=oth_head)
                Stdio.writeDatabase(oth_df, path_oth)

        # reset data
        self.pop_db, self.oth_db = [], []


    def outJudgement(self, condition_type:str, trial:int, total_evals:int=None):
        '''
            [sequential processing]
            function to judge output situation
            * When logging function call this function, must set to argument 'total_evals'
        '''
        judge = False
        if condition_type == 'loggingStandard' :
            if self.cnf.log['standard'] and (total_evals in self.slog_evals) :
                judge = True
        elif condition_type == 'loggingDetail' :
            # output only first trial
            if self.cnf.log['population']['trial'] == 'first-only' :
                if self.cnf.log['population']['out']and (total_evals in self.dlog_evals) and ( trial ==1 ) :
                    judge = True
            # output all trials
            elif self.cnf.log['population']['trial'] == 'all' :
                if self.cnf.log['population']['out']and (total_evals in self.dlog_evals) :
                    judge = True
            else :
                _trial = self.cnf.log['population']['trial']
                print(f'[LogData] Error: Do not exist dlog_trials {_trial} (class {self.__class__.__name__})')
                return
        elif condition_type == 'outLogStandard' :
            if self.cnf.log['standard']:
                judge = True
        elif condition_type == 'outLogDetail' :
            # output only first trial
            if self.cnf.log['population']['trial'] == 'first-only' :
                if self.cnf.log['population']['out']and  ( trial ==1 ) :
                    judge = True
            # output all trials
            elif self.cnf.log['population']['trial'] == 'all' :
                if self.cnf.log['population']['out']:
                    judge = True
            else :
                _trial = self.cnf.log['population']['trial']
                print(f'[LogData] Error: Do not exist dlog_trials {_trial} (class {self.__class__.__name__})')
                return
        else :
            print(f'[LogData] Error: Do not exist condition_type {condition_type} (class {self.__class__.__name__})')
            return
        return judge


    @staticmethod
    def _summarizeAllTrials(path_trial:str, path_result_trial:str, trial:int, index_evals:int, file:dict, prog_trial:re.compile, prog_dir_trial:re.compile, profile_report:bool=False):
        '''
            [batch processing]
            Get all standard log and output agrrigating log
        '''
        df_eval = []
        for filename in tqdm(gl.glob(path_trial), desc='Evaluation Loop'):
            _filename = os.path.basename(filename)
            evals = int(prog_trial.match(_filename).group(1))
            df_eval.append(Stdio.readDatabase(filename, index_col=0, header=None))
            df_eval[-1].loc[index_evals] = evals
        if not df_eval == []:
            df_trial = pd.concat(df_eval, axis=1).T
            df_trial = df_trial.set_index(index_evals).sort_index()
            # add basic statistics
            df_trial['FitnessValue'] = [ df_trial['PreviousFitness'].values[:row+1].max() for row in range(len(df_trial)) ]
            df_trial['NoVW'] = [ df_trial['NumberOfViolationWires'].values[:row+1].min() for row in range(len(df_trial)) ]
            df_trial['NoVL'] = [ df_trial['NumberOfViolationLayer'].values[:row+1].min() for row in range(len(df_trial)) ]

            # standard output
            _trial = int(prog_dir_trial.match(trial).group(1))
            _std_out = os.path.join(path_result_trial, file['trial'](_trial))
            Stdio.writeDatabase(df_trial, _std_out, index=True)
            print(f'[LogData] Output trial ({os.path.basename(_std_out)})')

            # profile output
            if(profile_report):
                from pandas_profiling import ProfileReport
                title = f'test ({trial})'
                profile = ProfileReport(df_trial, title=title, explorative=True)
                _profile_out = os.path.join(path_result_trial, file['profile'](_trial))
                profile.to_file(output_file=_profile_out)
                print(f'[LogData] Output profile report ({os.path.basename(_profile_out)})')


    @staticmethod
    def summarizeAllTrials(path_output_instance:str):
        '''
            [batch processing]
            Get all standard log and output agrrigating log (Parallel Processing)
        '''
        # Configuration
        path_output = os.path.dirname(path_output_instance)
        instance_name = os.path.basename(path_output_instance)
        path_log = os.path.dirname(path_output)
        path_root = os.path.dirname(path_log)
        path_result = os.path.join(path_root,'_result')
        path_result_instance = Stdio.makeDirectory(path_result, instance_name.replace('output','result'))
        dirname_trial = 'trial-*'
        filename_trial = 'out_gen*.csv'
        index_evals = 'FEs'

        file   = {
            'trial'     : lambda n: f'trial{n}_std.csv',
            'profile'   : lambda n: f'profile-report_trial{n}.html'
        }

        # Get value from _output
        pattern_trial = filename_trial.replace('*','(.*)')
        prog_trial = re.compile(pattern_trial)
        pattern_dir_trial = dirname_trial.replace('*','(.*)')
        prog_dir_trial = re.compile(pattern_dir_trial)

        # generate run sequences
        trial_queue, path_trial_queue, path_result_trial_queue = [], [], []
        for prob_name in os.listdir(path_output_instance):
            path_prob = os.path.join(path_output_instance, prob_name)
            path_result_trial_queue.extend([ Stdio.makeDirectory(path_result_instance, prob_name, 'trials') ]*len(os.listdir(path_prob)))
            for trial in os.listdir(path_prob):
                trial_queue.append(trial)
                path_trial_queue.append(os.path.join(path_prob, trial, filename_trial))

        # parallel process
        Parallel(n_jobs=-1)(
            [
                delayed(LogData._summarizeAllTrials)(
                    path_trial_queue[i], path_result_trial_queue[i], trial_queue[i], index_evals, file, prog_trial, prog_dir_trial, False)  for i in range(len(path_trial_queue)
                )
            ]
        )

        print(f'[LogData] Output summarizeAllTrials ({path_result_instance})')


class Stdio:
    '''
        Standard Input/Output library
    '''
    @staticmethod
    def moveWorkingDirectory():
        current_dir = os.getcwd()
        if os.path.split(current_dir)[1] == 'optimizer':
            os.chdir('..')

    @staticmethod
    def makeDirectory(directory:str, *directories:str, confirm=True) -> str:
        '''
            make interactively directory (deep make)
        '''
        # generate path
        path_out = os.path.join(directory, *directories)

        # generate directory for output
        if not os.path.isdir(path_out):
            os.makedirs(path_out)
        else :
            if confirm:
                print('＜＜ output folder conflict ＞＞')
                print(f'Folder name ({directories[-1]}) has already existed.')
                # overwrite or not
                overwrite, new_dir_out = None, None
                while overwrite is None :
                    overwrite = input('Do you delete existing data? (y/n) : ')
                    if not overwrite in ['y','n'] :
                        print(f'Invalid input. ( {overwrite} )\n')
                        overwrite = None

                # 1: overwrite (delete conventional data)
                if overwrite == 'y' :
                    st.rmtree(path_out)
                    os.makedirs(path_out)
                    # when you opened the folder in explorer, error happens not to access the folder.
                # 2: make new file
                elif overwrite == 'n' :
                    while new_dir_out is None :
                        new_dir_out = input('Please enter making folder name. : ')
                        # generate new path
                        new_path_out = os.path.join(directory, *directories[:-1], new_dir_out)
                        if os.path.isdir(new_path_out) :
                            print(f'Already existed. ( {new_dir_out} )\n')
                            new_dir_out = None
                    print(f'Valid folder name "{new_dir_out}".')
                    path_out = new_path_out
                    os.makedirs(path_out)
        return path_out


    @staticmethod
    def getNumberOfFiles(directory:str, file_template:str):
        '''
            get the number of files corresponding to condition

            Attributes
            ----------
            directory : path of directory
            file_template : file template  ( i.e., out_gen*.csv )
        '''
        path_template = os.path.join(directory, file_template)
        return len(gl.glob(path_template))


    @staticmethod
    def writeDatabase(df:pd.DataFrame, path_write:str, csv_or_xlsx:str=None, index:bool=False, header:bool=True) -> None:
        '''
            write Database (data frame) to csv/xlsx
        '''
        if csv_or_xlsx is None:
            # read extension
            csv_or_xlsx = path_write.split('.')[-1]
            assert csv_or_xlsx in ['csv','xlsx'], 'Error: Invalid file extension.'
        else:
            # add extension
            path_write = '{}.{}'.format(path_write,csv_or_xlsx)
        if csv_or_xlsx == 'xlsx':
            df.to_excel(path_write, engine='openpyxl', index=index, header=header)
        elif csv_or_xlsx == 'csv':
            df.to_csv(path_write, index=index, header=header)


    @staticmethod
    def readDatabase(path_read:str, index_col=0, header=0, names=None) -> pd.DataFrame:
        '''
            read Database (DataFrame) from csv/xlsx
            * You can get variable length csv as DataFrame when you assign optional parameters "names".
        '''
        csv_or_xlsx = path_read.split('.')[-1]
        if os.path.exists(path_read):
            if csv_or_xlsx == 'xlsx':
                df = pd.read_excel(path_read, index_col=index_col, header=header)
            elif csv_or_xlsx == 'csv':
                df = pd.read_csv(path_read, index_col=index_col, header=header, names=names)
        else:
            df = None
            raise FileNotFoundError(f'Do not exist {path_read}')
        return df


    @staticmethod
    def readDatabaseAsList(path_read:str) -> list:
        '''
            read Database (list) from csv
            * You can get variable length csv as list.
        '''
        if os.path.exists(path_read):
            with open(path_read) as f:
                reader = csv.reader(f)
                _list = [row for row in reader]
            return _list
        else:
            return None


    @staticmethod
    def getNumberOfColumns(path_read:str) -> pd.DataFrame:
        '''
            read numbers of columns from csv
            (for variable columns csv file)

            ex) n_col = Stdio.getNumberOfColumns(path_csv)
                col_name = [ 'name_{}'.format(i)  for i in range(n_col) ]
                df = Stdio.readDatabase(path_csv, names=col_name)
        '''
        if os.path.exists(path_read):
            cols = Stdio.readDatabaseAsList(path_read)
            return max([ len(col)  for col in cols ])
        else:
            return -1


    @staticmethod
    def saveExperimentalData(path_out:str, data_dict:dict, file_name=None, seed=0, csv_or_xlsx:str='csv', display='vertical'):
        '''
            save experimental data
        '''
        if file_name is None:
            file_name = f'trial{seed}_experimental-data'
        # make directory
        path_table = Stdio.makeDirectory(os.path.dirname(path_out),os.path.basename(path_out),confirm=False)
        file_name = file_name if file_name.split('.')[-1] in ['csv','xlsx'] else f'{file_name}.{csv_or_xlsx}'
        path_table = os.path.join(path_table, file_name)
        if display == 'horizontal':
            df = pd.DataFrame(data_dict.values(), index=data_dict.keys())
            Stdio.writeDatabase(df,path_table,index=True,header=None)
        elif display == 'vertical':
            df = pd.DataFrame(data_dict.values(), index=data_dict.keys()).T
            Stdio.writeDatabase(df,path_table)
        else:
            raise AttributeError(f'invalid attribute "display" = "{display}" in Stdio.saveExperimenatalData()')

    @staticmethod
    def drawFigure(x, y=None, y2=None, y_q25=None, y_q75=None, figsize:tuple=(10,4), title=None, label='line-1', label2='line-2', xlim=None, ylim=None, ylim2=None, xlabel=None, ylabel=None, ylabel2=None, xscale:str='linear',yscale:str='linear', grid:bool=False, grid2:bool=False, legend:bool=False, linestyle='-', color='blue', color2='orange', cmap=None, cmap_lim=None, cmap2=None, cmap2_lim=None, option:str='', option2:str='',colorbar:bool=False, draw_type='plot', show:bool=False, save:bool=True, path_out=None, dpi:int=300, output:bool=False):
        '''
            draw plot/scatter figure

            - x : list(int,float,str)
            - y, y2, y_q25, y_q75 : list(int,float), nd.array(1d,2d), pd.core.series.Series, None
            - figsize : tuple(int,float)
            - title, xlabel, ylabel, ylabel2 : str, None
            - label, label2 : str, list(str)
            - xlim, ylim, ylim2, cmap_lim, cmap2_lim : tuple(int,float), list(int,float), nd.array(1d), None
            - xscale, yscale : str [*1]
            - grid, grid2, legend, colorbar, show, save, output: bool
            - linestyle : str, list(str) [*2]
            - color, color2 : str, list(str) [*3]
            - cmap, cmap2 : colormap object or None [*4]
            - option, option2 : str  [original argument]
            - draw_type : str, list(str)  [*** of ax.***(~)]
            - dpi : int
            - path_out : str, None

            [*1] https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.xscale.html
            [*2] https://matplotlib.org/3.3.3/gallery/lines_bars_and_markers/linestyles.html
            [*3] https://pythondatascience.plavox.info/matplotlib/%E8%89%B2%E3%81%AE%E5%90%8D%E5%89%8D
            [*4] https://matplotlib.org/api/cm_api.html#matplotlib.cm.get_cmap
                 https://matplotlib.org/examples/color/colormaps_reference.html

        '''
        x = np.array(x)
        y = None if y is None else np.array(y)
        option = ',{}'.format(option) if option != '' else option
        option2 = ',{}'.format(option2) if option2 != '' else option2
        # create figure and axis(axes)
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax2, sm = None, None
        # draw figure in axis
        if y is None:
            draw_type = 'bar'
            _color = [color]*len(x) if isinstance(color, str) else color
            assert isinstance(_color, str), 'Error: invalid color {}.'.format(_color)
            ax.bar(x, color=_color)
        elif y.ndim == 1:
            assert len(x)==len(y), 'Error: different between x length {} and y length {}.'.format(len(x), len(y))
            assert isinstance(linestyle, str), 'Error: invalid linestyle {}.'.format(linestyle)
            assert isinstance(color, str), 'Error: invalid color {}.'.format(color)
            assert isinstance(label, str), 'Error: invalid label {}'.format(label)
            eval('ax.{}(x, y, linestyle=linestyle, color=color, label=label {})'.format(draw_type, option))
        elif y.ndim == 2:
            _linestyle = [linestyle]*y.shape[0] if isinstance(linestyle, str) else linestyle
            _color = [color]*y.shape[0] if isinstance(color, str) else color
            _label = [label]*y.shape[0] if isinstance(label, str) else label
            if not cmap is None:
                _color = [cmap(k/y.shape[0])  for k in range(y.shape[0])]
                if not cmap_lim is None:
                    norm = cls.Normalize(vmin=cmap_lim[0], vmax=cmap_lim[1])
                    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            _x = np.tile(x,(y.shape[0],1)) if x.ndim == 1 else x
            assert len(_linestyle)==y.shape[0], 'Error: invalid linestyle {}.'.format(_linestyle)
            assert len(_color)==y.shape[0], 'Error: invalid color {}.'.format(_color)
            assert len(_label)==y.shape[0], 'Error: invalid label {}.'.format(_label)
            assert _x.shape==y.shape, 'Error: invalid x shape {}.'.format(_x.shape)
            for _i in range(y.shape[0]):
                if _i==0 or y2 is None:
                    # label
                    eval('ax.{}(_x[_i], y[_i], linestyle=_linestyle[_i], color=_color[_i], label=_label[_i] {})'.format(draw_type, option))
                else:
                    # label-less
                    eval('ax.{}(_x[_i], y[_i], linestyle=_linestyle[_i], color=_color[_i] {})'.format(draw_type, option))
        else:
            print('Error: invalid y shape {}.'.format(y.shape), file=sys.stderr)

        if isinstance(y_q25, (list,np.ndarray, pd.core.series.Series)) and isinstance(y_q75, (list,np.ndarray, pd.core.series.Series)):
            alpha = 0.1
            # fill_between
            y_q25, y_q75 = np.array(y_q25), np.array(y_q75)
            if y_q25.ndim == 1 and y_q75.ndim == 1 and y_q25.size != 0 and y_q75.size != 0:
                ax.fill_between(x, y_q25, y_q75, facecolor=color, alpha=alpha)
            elif y_q25.ndim == 2 and y_q75.ndim == 2:
                for _i in range(y.shape[0]):
                    ax.fill_between(x[_i], y_q25[_i], y_q75[_i], facecolor=color[_i], alpha=alpha)
            else:
                print('Error: invalid y_q25 shape {} or y_q75 shape {}.'.format(y_q25.shape, y_q75.shape), file=sys.stderr)

        if isinstance(y2, (list, np.ndarray, pd.core.series.Series)):
            # twin axes mode
            y2 = np.array(y2)
            ax2 = ax.twinx()
            draw_type = [draw_type]*2 if isinstance(draw_type, str) else draw_type
            if y2.ndim == 1:
                assert len(x)==len(y2), 'Error: different between x length {} and y2 length {}.'.format(len(x), len(y2))
                assert isinstance(linestyle, str), 'Error: invalid linestyle {}.'.format(linestyle)
                assert isinstance(color2, str), 'Error: invalid color {}.'.format(color2)
                assert isinstance(label2, str), 'Error: invalid label {}'.format(label)
                eval('ax2.{}(x, y2, linestyle=linestyle, color=color2, label=label2 {})'.format(draw_type[1], option2))
            elif y2.ndim == 2:
                _linestyle = [linestyle]*y2.shape[0] if isinstance(linestyle, str) else linestyle
                _color2 = [color2]*y2.shape[0] if isinstance(color2, str) else color2
                _label2 = [label2]*y.shape[0] if isinstance(label2, str) else label2
                if not cmap2 is None:
                    _color2 = [cmap2(k/y.shape[0])  for k in range(y2.shape[0])]
                    if not cmap2_lim is None:
                        norm = cls.Normalize(vmin=cmap2_lim[0], vmax=cmap2_lim[1])
                        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap2)
                _x = np.tile(x,(y2.shape[0],1)) if x.ndim == 1 else x
                assert len(_linestyle)==y2.shape[0], 'Error: invalid linestyle {}.'.format(_linestyle)
                assert len(_color2)==y2.shape[0], 'Error: invalid color {}.'.format(_color2)
                assert len(_label2)==y.shape[0], 'Error: invalid label2 {}.'.format(_label2)
                assert _x.shape==y2.shape, 'Error: invalid x shape {}.'.format(_x.shape)
                for _i in range(y2.shape[0]):
                    if _i==0 or y is None:
                        # label
                        eval('ax2.{}(_x[_i], y2[_i], linestyle=linestyle[_i], color=color2[_i], label=label2[_i] {})'.format(draw_type[1],option2))
                    else:
                        # label-less
                        eval('ax2.{}(_x[_i], y2[_i], linestyle=linestyle[_i], color=color2[_i] {})'.format(draw_type[1],option2))
            else:
                print('Error: invalid y2 shape {}.'.format(y2.shape), file=sys.stderr)

        if not title is None:
            assert isinstance(title, str), 'Error: invalid title {}.'.format(title)
            ax.set_title(title)
        if not xlim is None:
            assert isinstance(xlim, (tuple, list, np.ndarray, int, float)), 'Error: invalid xlim {}.'.format(xlim)
            ax.set_xlim(xlim)
        if not ylim is None:
            assert isinstance(ylim, (tuple, list, np.ndarray, int, float)), 'Error: invalid ylim {}.'.format(ylim)
            ax.set_ylim(ylim)
        if (not ylim2 is None) and y.ndim == 2:
            assert isinstance(ylim, (tuple, list, np.ndarray, int, float)), 'Error: invalid ylim2 {}.'.format(ylim2)
            ax2.set_ylim(ylim2)
        if not xlabel is None:
            assert isinstance(xlabel, str), 'Error: invalid xlabel {}.'.format(xlabel)
            ax.set_xlabel(xlabel)
        if not ylabel is None:
            assert isinstance(ylabel, str), 'Error: invalid ylabel {}.'.format(xlabel)
            ax.set_ylabel(ylabel)
        if not ylabel2 is None:
            assert isinstance(ylabel2, str), 'Error: invalid ylabel2 {}.'.format(xlabel)
            ax2.set_ylabel(ylabel2)
        if not xscale is None:
            assert isinstance(xscale, str), 'Error: invalid xscale {}.'.format(xscale)
            ax.set_xscale(xscale)
        if not yscale is None:
            assert isinstance(yscale, str), 'Error: invalid yscale {}.'.format(yscale)
            ax.set_yscale(yscale)
        ax.grid(grid)
        if not y2 is None:
            ax2.grid(grid2)
        if legend:
            if not y2 is None and not ylabel2 is None:
                handler1, label1 = ax.get_legend_handles_labels()
                handler2, label2 = ax2.get_legend_handles_labels()
                ax.legend(handler1 + handler2, label1 + label2, loc=2, borderaxespad=0.)
            else:
                ax.legend()
        if colorbar:
            fig.colorbar(sm)
        if show:
            plt.show()
        if save:
            assert isinstance(path_out, str), 'Error: invalid path_out {}'.format(path_out)
            fig.savefig(path_out, dpi=dpi)
        if output:
            try:
                return fig,ax,ax2
            except:
                return fig,ax
        plt.close()


    @staticmethod
    def outputTracebackError(path_out='.', error_content=None):
        '''
            output traceback error
        '''
        file_encoding   = 'UTF-8'
        computer_name = pf.node()
        error_path_out  = os.path.join(path_out, f'error_main_{computer_name}.txt')
        head            = '+'*10 + ' < Error Factor > ' + '+'*10 + '\n'
        error_content = error_content.decode(encoding=file_encoding) if isinstance(error_content, bytes) else error_content
        error_content = tb.format_exc() if error_content is None else error_content
        body = [head, error_content]
        # confirm path
        if os.path.isfile(error_path_out):
                os.remove(error_path_out)
        # output file
        with open( error_path_out , 'x', encoding=file_encoding ) as f :
            f.writelines('\n'.join(body))


if __name__ == "__main__":
    #-------------------------------
    # Summarize trials from _output
    #-------------------------------
    paths = [
        r"C:\Users\taichi\Documents\SourceCode\AQFPLogic-Optimization\_log\_output\output_ver2.2_test_100k-evals_11-trial_CCDO-EG2_2021-05-18_04-43-52",
        r"C:\Users\taichi\Documents\SourceCode\AQFPLogic-Optimization\_log\_output\output_ver2.2_test_100k-evals_11-trial_CCDO-EG4_2021-05-18_06-24-14"
    ]
    for path in paths:
        LogData.summarizeAllTrials(path)

