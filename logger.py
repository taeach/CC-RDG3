# Data Logger
# version 1.1 (2022/01/12)

import  os                       # standard library
import  re                       # standard library
import  glob        as gl       # standard library
import  sys                      # standard library
import  time        as tm       # standard library
import  platform    as pf       # standard library
import  subprocess  as sp       # standard library
import  types       as tp       # standard library
import  numpy       as np
import  pandas      as pd
import  yaml
from    copy        import deepcopy
from    tqdm        import tqdm
from    joblib      import Parallel, delayed
from    utils       import Stdio, log
''' For Annotations '''
from config         import Configuration


class DataLogger:
    def __init__(self, cnf:Configuration, prob_name:str=None):

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


    def addExeTime(self, exe_time:int):
        '''Add exe-time counter
        '''
        self.exe_time += exe_time


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



    def logging(self, opt:Configuration, total_evals:int, trial:int) :
        '''
            Organization function to summarize logging function
        '''
        # get problem dimension
        self.prob_dim = len(opt.pop.getBest[0])
        self.loggingStandard(opt, total_evals, trial, option='no-sort')
        self.loggingDetail(opt, total_evals, trial)


    def outLog(self, opt:Configuration, total_evals:int, trial:int) :
        '''
            Organization function to summarize log-output function
        '''
        self.outLogStandard(opt, total_evals, trial, option='no-sort')
        self.outLogDetail(opt, total_evals, trial)


    def loggingStandard(self, opt:Configuration, total_evals:int, trial:int, option:str='no-sort'):
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


    def loggingDetail(self, opt:Configuration, total_evals:int, trial:int):
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


    def outLogStandard(self, opt:Configuration, total_evals:int, trial:int, option:str='no-sort'):
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


    def outLogDetail(self, opt:Configuration, total_evals:int, trial:int):
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
                delayed(DataLogger._summarizeAllTrials)(
                    path_trial_queue[i], path_result_trial_queue[i], trial_queue[i], index_evals, file, prog_trial, prog_dir_trial, False)  for i in range(len(path_trial_queue)
                )
            ]
        )

        print(f'[LogData] Output summarizeAllTrials ({path_result_instance})')



if __name__ == "__main__":
    #-------------------------------
    # Summarize trials from _output
    #-------------------------------
    paths = [
        r"C:\Users\taichi\Documents\SourceCode\AQFPLogic-Optimization\_log\_output\output_ver2.2_test_100k-evals_11-trial_CCDO-EG2_2021-05-18_04-43-52",
        r"C:\Users\taichi\Documents\SourceCode\AQFPLogic-Optimization\_log\_output\output_ver2.2_test_100k-evals_11-trial_CCDO-EG4_2021-05-18_06-24-14"
    ]
    for path in paths:
        DataLogger.summarizeAllTrials(path)

