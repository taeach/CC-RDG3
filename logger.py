# Data Logger
# version 1.4 (2022/01/13)

import os
import sys
import time         as tm
import platform     as pf
import subprocess   as sp
import types        as tp
from typing         import Callable
import numpy        as np
import pandas       as pd
import yaml
from copy           import deepcopy
from utils          import Stdio, log
''' For Function Annotations '''
from config         import Configuration


class DataLogger:
    def __init__(self, cnf:Configuration, prob_name:str=None):

        self.cnf                = cnf               # Configuration class
        self.prob_name          = prob_name         # problem to solve
        self.prob_dim           = None              # problem dimension
        self.config             = None              # output config.yml
        self.std_db             = []                # standard database
        self.pop_db             = []                # population database
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
        if self.cnf.log['standard']['out']:
            if not os.path.isdir(self.path_trial):
                os.makedirs(self.path_trial)


    def startStopwatch(self) -> None:
        '''Start Stopwatch for exe-time counter
        '''
        if self.timer is None:
            self.timer = tm.mktime(tm.localtime())
        else:
            log(self, f'Error: Try again after stopping the stopwatch. [startStopwatch -> startStopwatch]', output=sys.stderr)


    def stopStopwatch(self) -> None:
        '''Stop Stopwatch for exe-time counter
        '''
        if not self.timer is None:
            self.exe_time += int(tm.mktime(tm.localtime()) - self.timer)
            self.timer = None
        else:
            log(self, f'Error: Try again after starting the stopwatch. [ ? -> stopStopwatch]', output=sys.stderr)


    def addExeTime(self, exe_time:int) -> None:
        '''Add exe-time counter
        '''
        self.exe_time += exe_time


    def outSetting(self, timing:str, stdout:bool=True, timer:bool=True) -> None:
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
                    log(self, f'\n{config_yml}')

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
                    log(self, f'* Execution time : {self.total_exe_time}[sec]')
            else :
                raise ValueError(f'Invalid argument timing={timing} in LogData.outSetting()')

    ''' Serial Process '''
    def logging(self, opt:Callable, total_evals:int, trial:int) :
        '''Log to database
        '''
        if self.prob_dim is None:
            self.prob_dim = opt.fnc.prob_dim
        self.loggingStandard(opt, total_evals, trial)
        self.loggingAdvanced(opt, total_evals, trial)


    def outLog(self, opt:Callable, total_evals:int, trial:int) :
        '''Output log database
        '''
        self.outLogStandard(opt, total_evals, trial)
        self.outLogAdvanced(opt, total_evals, trial)


    def loggingStandard(self, opt:Callable, total_evals:int, trial:int):
        '''Get standard log
        '''
        if self.outJudgement('loggingStandard', trial, total_evals) :
            # Result:
            f_best = opt.pop.getBest[1]
            f_new = opt.pop.getCurrent[1]
            dtset = [total_evals, f_new, f_best]
            self.std_db.append(dtset)

            # Regular-log:
            if total_evals in self.regular_evals:
                std_head = ['FEs', 'Fitness', 'BestFitness']
                # slice FEs
                count = self.regular_evals.index(total_evals)
                start_FEs = self.regular_evals[count-1] if count!=0 else 0
                end_FEs = self.regular_evals[count]
                std_df = pd.DataFrame(self.std_db, columns=std_head)
                dif_std_df = std_df.query(f'{start_FEs} < FEs <= {end_FEs}')
                path_log_dif = os.path.join(self.path_trial, self.cnf.filename['regular-log'](trial,count+1))
                Stdio.writeDatabase(dif_std_df, path_log_dif)


    def outLogStandard(self, opt:Callable, total_evals:int, trial:int):
        '''Output standard log
        '''
        # Result:
        if self.outJudgement('outLogStandard', trial) :
            std_head = ['FEs', 'Fitness', 'BestFitness']
            std_df = pd.DataFrame(self.std_db, columns=std_head)
            path_std = os.path.join(self.path_trial, self.cnf.filename['result'](trial))
            Stdio.writeDatabase(std_df, path_std)
        # Standard-output:
        f_best = opt.pop.getBest[1]
        f_new = opt.pop.getCurrent[1]
        ProbName = self.prob_name.ljust(13)
        Trial = str(trial).ljust(self.trial_digit)
        FEs = str(total_evals).rjust(self.evals_digit)
        Fitness = str(f_new)[:self.fitness_digit].ljust(self.fitness_digit)
        BestFitness = str(f_best)[:self.fitness_digit].ljust(self.fitness_digit)
        log(f'{ProbName}: trial-{Trial}',
            f'FEs = {FEs}  |  Fitness = {Fitness}  |  BestFitness = {BestFitness}')
        # Reset database
        self.std_db= []


    def loggingAdvanced(self, opt:Callable, total_evals:int, trial:int):
        '''Get advanced log
        '''
        if self.outJudgement('loggingAdvanced', trial, total_evals) :
            # Result-pop:
            if self.cnf.log['population']['out']:
                x, f = opt.pop.getPopulation
                for i,(xi,fi) in enumerate(zip(x,f)):
                    # Calculate Diversity Measure
                    xi_mean = np.sum(xi,axis=0) / self.cnf.max_pop
                    div_xi = np.sum(np.sum((xi - xi_mean)**2, axis=1)**(1/2))
                    for j,(xij,fij) in enumerate(zip(xi,fi)):
                        self.pop_db.append([total_evals, i, j, fij, div_xi] + list(xij))


    def outLogAdvanced(self, opt:Callable, total_evals:int, trial:int):
        '''Output advanced log
        '''
        if self.outJudgement('outLogAdvanced', trial) :
            # Result-pop:
            if self.cnf.log['population']['out']:
                pop_head = ['FEs', 'div', 'pop', 'Fitness', 'Diversity']
                x_head = [f'x{i}' for i in range(self.prob_dim)]
                pop_head += x_head
                pop_df = pd.DataFrame(self.pop_db, columns=pop_head)
                path_pop = os.path.join(self.path_trial, self.cnf.filename['result-pop'](trial))
                Stdio.writeDatabase(pop_df, path_pop)
                # Profile Report
                if self.cnf.log['population']['report']:
                    from pandas_profiling import ProfileReport
                    title = f'Analysis Report (trial{trial})'
                    profile = ProfileReport(pop_df, title=title, explorative=True)
                    path_profile = os.path.join(self.path_trial, self.cnf.filename['profile-report'])
                    profile.to_file(output_file=path_profile)
                    log('DataLogger', f'Output profile report ({os.path.basename(path_profile)})')
        # Reset database
        self.pop_db, self.pop_db = [], []


    def outJudgement(self, condition_type:str, trial:int, total_evals:int=None):
        '''Judge output situation
        * When logging function call this function, must set to argument 'total_evals'
        '''
        judge = False
        if condition_type == 'loggingStandard' :
            if self.cnf.log['standard']['out'] and (total_evals in self.slog_evals) :
                judge = True
        elif condition_type == 'loggingAdvanced' :
            # output only first trial
            if self.cnf.log['population']['trial'] == 'first-only' :
                if self.cnf.log['population']['out'] and (total_evals in self.dlog_evals) and ( trial ==1 ) :
                    judge = True
            # output all trials
            elif self.cnf.log['population']['trial'] == 'all' :
                if self.cnf.log['population']['out'] and (total_evals in self.dlog_evals) :
                    judge = True
            else :
                _trial = self.cnf.log['population']['trial']
                log(self, f'Error: Do not exist dlog_trials {_trial}')
                return
        elif condition_type == 'outLogStandard' :
            if self.cnf.log['standard']['out']:
                judge = True
        elif condition_type == 'outLogAdvanced' :
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
                log(self, f'Error: Do not exist dlog_trials {_trial}')
                return
        else :
            log(self, f'Error: Do not exist condition_type {condition_type}')
            return
        return judge

    ''' Batch Process '''
    def loggingSummary(self, opt:Callable, trial:int) :
        '''Log Summary to database
        '''
        if self.prob_dim is None:
            self.prob_dim = opt.fnc.prob_dim
        f_best = opt.init_fitness
        # try:
        for total_evals,f_new in enumerate(opt.f_log,start=1):
            if opt.superior(f_new,f_best):
                f_best = f_new
            self.loggingSummaryStandard(opt, total_evals, trial, f_new, f_best)
            self.loggingSummaryAdvanced(opt, total_evals, trial, f_new, f_best)
        # except AttributeError:
        #     pass


    def loggingSummaryStandard(self, opt:Callable, total_evals:int, trial:int, f_new:float, f_best:float) :
        '''Get Standard Log Summary
        '''
        if self.outJudgement('loggingStandard', trial, total_evals) :
            # Result:
            dtset = [total_evals, f_new, f_best]
            self.std_db.append(dtset)
            # Regular-log:
            if total_evals in self.regular_evals:
                std_head = ['FEs', 'Fitness', 'BestFitness']
                # slice FEs
                count = self.regular_evals.index(total_evals)
                start_FEs = self.regular_evals[count-1] if count!=0 else 0
                end_FEs = self.regular_evals[count]
                std_df = pd.DataFrame(self.std_db, columns=std_head)
                dif_std_df = std_df.query(f'{start_FEs} < FEs <= {end_FEs}')
                path_log_dif = os.path.join(self.path_trial, self.cnf.filename['regular-log'](trial,count+1))
                Stdio.writeDatabase(dif_std_df, path_log_dif)


    def loggingSummaryAdvanced(self, opt:Callable, total_evals:int, trial:int, f_new:float, f_best:float):
        '''Get advanced log summary
        '''
        if self.cnf.log['population']['out']:
            if self.outJudgement('loggingAdvanced', trial, total_evals) :
                # Result-pop:
                i,j,div_xi = 0,0,0
                xij = [np.nan]
                self.pop_db.append([total_evals, i, j, f_new, div_xi] + list(xij))
