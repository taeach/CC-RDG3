# Run Optimizer
# version 10.5 (2021/12/1)

import time             as tm
from typing             import Any
import numpy            as np
from joblib             import Parallel, delayed
from tqdm               import trange
import config           as cf
import function         as fc
import optimizer        as op
import logger           as lg
import dataprocessing   as dp
import itertools        as it

''' Run optimizer (1-trial) '''
def runOpt(opt:Any, cnf:cf.Configuration, fnc:fc.Function, log:lg.LogData, plt:dp.Plot, j:int=1) -> None:
    # set the seed value of the random number
    cnf.setRandomSeed(seed=j)
    # initialize optimizer
    for k in range(cnf.max_pop):
        log.startStopwatch()
        opt.initialize(opt.blankOpt(fnc.total_evals))
        log.stopStopwatch()
        log.logging(opt, fnc.total_evals, j)
        plt.getDataX(opt.pop.getCurrent[0], fnc.total_evals)
    # update optimizer
    while fnc.total_evals < cnf.max_evals :
        log.startStopwatch()
        opt.update(opt.blankOpt(fnc.total_evals))
        log.stopStopwatch()
        log.logging(opt, fnc.total_evals,j)
        plt.getDataX(opt.pop.getCurrent[0], fnc.total_evals)

    log.outLog(opt, fnc.total_evals, j)
    plt.plotFigureAnimation()
    return log.exe_time


''' Run optimizer for Code Performance Profiler (1-trial) '''
def performanceChecker() -> None:
    import os
    from line_profiler import LineProfiler
    # overwrite config for performance checker
    cnf = cf.Configuration()
    cnf.prob_name = ['8-bitAdder', '16-bitAdder']
    cnf.initial_seed  = 1
    cnf.max_trial   = 1
    cnf.max_evals   = 1000
    cnf.comment = f'_performance-check-mode_{cnf.time}'
    path_analysis = Stdio.makeDirectory(cnf.path_out, '_code_analysis', confirm=False)

    log_settings = lg.LogData(cnf)
    log_settings.outSetting('start')

    for i in trange(len(cnf.prob_name), desc='Problem Loop'):
        log = lg.LogData(cnf, cnf.prob_name[i])
        for j in trange(cnf.initial_seed, cnf.max_trial+cnf.initial_seed, desc='Trial Loop'):
            fnc = fc.Function(cnf, cnf.prob_name[i], j)
            opt = eval('op.{}(cnf, fnc)'.format(cnf.opt_name.replace('-','')))
            plt = dp.Plot(cnf, fnc)
            prf1, prf2 = LineProfiler(), LineProfiler()
            # set the seed value of the random number
            cnf.setRandomSeed(seed=j)
            # initialize optimizer
            prf1.add_function(opt.initialize)
            for k in range(cnf.max_pop):
                prf1.runcall(opt.initialize, opt.blankOpt(fnc.total_evals))
                # opt.initialize(opt.blankOpt(fnc.total_evals))
                log.logging(opt, fnc.total_evals)
                plt.getDataX(opt.pop.getCurrent[0], fnc.total_evals)
            # log output
            with open(os.path.join(path_analysis, f'{cnf.opt_name}_initialize_ver{cnf.version}_{cnf.prob_name[i]}.log'), 'w') as f:
                prf1.print_stats(stream=f)
            # update optimizer
            prf2.add_function(opt.update)
            while fnc.total_evals < cnf.max_evals :
                prf2.runcall(opt.update, opt.blankOpt(fnc.total_evals))
                log.logging(opt, fnc.total_evals)
                plt.getDataX(opt.pop.getCurrent[0], fnc.total_evals)

            log.outLog(opt, fnc.total_evals)
            plt.plotFigureAnimation()
            # log output
            with open(os.path.join(path_analysis, f'{cnf.opt_name}_update_ver{cnf.version}_{cnf.prob_name[i]}.log'), 'w') as f:
                prf2.print_stats(stream=f)
            del opt,plt, prf1, prf2
        sts = dp.Statistics(cnf, fnc, log.path_out, log.path_trial)
        sts.outStatistics()
        del log,fnc,sts

    log_settings.outSetting('end')
    cnf.deleteFolders()
    del log_settings, cnf


def runAll() -> None:
    ''' Main Process for Debug (Series) for One Parameter
    '''
    cnf = cf.Configuration()
    log_settings = lg.LogData(cnf)
    log_settings.outSetting('start',timer=False)
    exe_time = []
    opt_name = cnf.opt_name.replace('-','')
    optimizer = eval(f'op.{opt_name}')

    for i in trange(len(cnf.prob_name), desc='Problem Loop'):
        log = lg.LogData(cnf, cnf.prob_name[i])
        for j in trange(cnf.initial_seed, cnf.max_trial+cnf.initial_seed, desc='Trial Loop'):
            fnc = fc.Function(cnf, cnf.prob_name[i], j)
            opt = optimizer(cnf,fnc)
            plt = dp.Plot(cnf, fnc)
            exe_time.append(runOpt(opt, cnf, fnc, log, plt, j))
            del opt,plt
        dp.Statistics(cnf, fnc, log.path_out, log.path_trial).outStatistics()
        del log,fnc

    log_settings.total_exe_time = int(np.array(exe_time).sum())
    log_settings.average_exe_time = int(np.average(exe_time))
    log_settings.outSetting('end',timer=False)
    cnf.deleteFolders()
    del log_settings, cnf


def runParallel(order:str='trial'):
    '''Run program by parallel process (Multi-Parameters)

    Args:
        order (str, optional): Order to run queue ('trial' or 'problem'). Defaults to 'trial'.
    '''
    root_cnf = cf.Configuration()
    root_cnf.addComment(f'root')
    log_settings = lg.LogData(root_cnf)
    log_settings.outSetting('start')

    # detect loop params
    no_loop_list = ['prob_name']
    loop_params = {}
    for var_name,var_value in vars(root_cnf).items():
        if isinstance(var_value, list) and not var_name in no_loop_list:
            loop_params[var_name] = var_value.copy()
    loop_param_key = tuple(loop_params.keys())
    loop_num = len(list(it.product(*loop_params.values())))

    print('[Parallel] Generate all instances')
    # generate all instances
    opt_name = root_cnf.opt_name.replace('-','')
    optimizer = eval(f'op.{opt_name}')
    fncs, cnfs, opts, plts, logs = [], [], [], [], []
    # set looper
    if loop_num == 1:
        looper = [None]
    else:
        looper = it.product(*loop_params.values())
    for i,loop_param in enumerate(looper):
        _cnf = cf.Configuration()
        if not loop_param is None:
            # time delay for folder name conflict
            tm.sleep(1)
            for var_name,var_value in zip(loop_param_key,loop_param):
                if isinstance(var_value,str):
                    exec(f'_cnf.{var_name} = "{var_value}"')
                else:
                    exec(f'_cnf.{var_name} = {var_value}')
                _cnf.addComment(f'{var_name[:2]}={var_value}')
        cnfs.append(_cnf)
        fncs.append([])
        opts.append([])
        plts.append([])
        logs.append([])
        for j in range(len(root_cnf.prob_name)):
            logs[i].append(lg.LogData(cnfs[i], cnfs[i].prob_name[j]))
            fncs[i].append([])
            opts[i].append([])
            plts[i].append([])
            for k,seed in enumerate(range(root_cnf.initial_seed, root_cnf.max_trial+root_cnf.initial_seed)):
                if loop_num == 1:
                    fncs[i][j].append(fc.Function(cnfs[i], cnfs[i].prob_name[j], trial=seed))
                else:
                    fncs[i][j].append(fc.Function(cnfs[i], cnfs[i].prob_name[j], trial=seed, param_name=str(loop_param)))
                opts[i][j].append(optimizer(cnfs[i], fncs[i][j][k]))
                plts[i][j].append(dp.Plot(cnfs[i], fncs[i][j][k]))

    # generate run queue
    print('[Parallel] Generate run queue')
    if order=='trial':
        ''' prob-1,trial-1 -> prob-1,trial-2 -> prob-1,trial-3 ->... '''
        instance_queue   = np.repeat(np.arange(len(root_cnf.prob_name)), root_cnf.max_trial*loop_num)
        trial_queue = np.tile(np.repeat(np.arange(root_cnf.max_trial), loop_num),len(root_cnf.prob_name))
        param_queue = np.tile(np.arange(loop_num), len(root_cnf.prob_name)*root_cnf.max_trial)
    elif order=='problem':
        ''' prob-1,trial-1 -> prob-2,trial-1 -> prob-3,trial-1 ->... '''
        instance_queue   = np.tile(np.arange(len(root_cnf.prob_name)), root_cnf.max_trial*loop_num)
        trial_queue = np.tile(np.repeat(np.arange(root_cnf.max_trial), len(root_cnf.prob_name)),loop_num)
        param_queue = np.repeat(np.arange(loop_num), len(root_cnf.prob_name)*root_cnf.max_trial)

    # start process
    print('[Parallel] Main Process Begins!')
    for i in range(loop_num):
        logs[i][0].outSetting('start',stdout=False,timer=False)

    # parallel process
    exe_time = Parallel(n_jobs=root_cnf.n_jobs)( [delayed(runOpt)(opts[p][n][trial], cnfs[p], fncs[p][n][trial], logs[p][n], plts[p][n][trial], root_cnf.initial_seed+trial)  for trial,n,p in zip(trial_queue,instance_queue,param_queue) ] )

    # end process
    for i in range(loop_num):
        _exe_time = np.array(exe_time)[param_queue==i]
        logs[i][0].total_exe_time = int(_exe_time.sum())
        logs[i][0].average_exe_time = int(np.average(_exe_time))
        logs[i][0].outSetting('end',stdout=False,timer=False)
    del opts,plts
    print('[Parallel] Main Process Finished!')

    # statistics process
    for i in range(loop_num):
        for j in range(len(root_cnf.prob_name)):
            dp.Statistics(cnfs[i], fncs[i][j][0], logs[i][j].path_out, logs[i][j].path_trial).outStatistics()
    del logs,fncs

    # termination process
    for i in range(loop_num):
        cnfs[i].deleteFolders()
    log_settings.outSetting('end')
    root_cnf.deleteFolders('all')
    del root_cnf, log_settings
    print('[Parallel] All Process Finished!')


''' main '''
if __name__ == '__main__':
    # (1) working directory movement
    from logger import Stdio
    Stdio.moveWorkingDirectory()
    # (2) run
    runParallel('problem')
    # runAll()
    # (3) analyze
    # performanceChecker()