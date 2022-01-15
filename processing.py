# Data Processing
# version 1.3 (2022/01/15)

import os
import sys
from typing             import Any
import numpy            as np
import pandas           as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib         import pyplot as plt
from sklearn.manifold   import TSNE
import networkx         as nx
from utils              import Stdio, log
''' For Function Annotations '''
from config             import Configuration
from function           import Function
from logger             import DataLogger


class DataProcessing:
    def __init__(self, cnf:Configuration, fncs:list[Function], dlg:DataLogger, opts:list[Any]):
        self.cnf        = cnf
        self.fncs       = fncs
        self.fnc        = fncs[0]
        self.dlg        = dlg
        self.opts       = opts
        self.path_out   = dlg.path_out
        self.path_out_parent   = os.path.split(self.path_out)[0]
        self.path_dt    = dlg.path_trial
        self.prob_name  = self.fnc.prob_name
        self.cmap_10    = plt.get_cmap('tab10')
        self.cmap_cont  = plt.get_cmap('RdYlBu')
        self.trial      = 1
        # Graph fonts
        plt.rcParams['font.family'] = 'Times New Roman'
        self.performances = ('BestFitness', 'Fitness')


    def outProcessing(self) -> None:
        '''Make statistics process and visualization from the standard and detail database
        '''
        if self.cnf.log['standard']['out']:
            self.aggregateTrials()
            self.outStandardProcessing()
            self.outAdditionalProcessing()

    def aggregateTrials(self) -> None:
        '''Make aggregation file from the standard database (Independent Process)
        '''
        # Count result file
        max_trial = Stdio.getNumberOfFiles(self.path_dt,self.cnf.filename['result']('*'))
        # Get all trials result files
        df = {}
        self.path_trials = {}
        for i in range(self.cnf.initial_seed, max_trial+self.cnf.initial_seed):
            # Extract database from trialN_standard.csv
            filename = self.cnf.filename['result'](i)
            path_dt = os.path.join(self.path_dt, filename)
            dtset = Stdio.readDatabase(path_dt)
            for pfm in self.performances:
                col_name = filename.split('.')[0]
                # Make database
                if i == 1:
                    # make dataframe
                    df[pfm] = pd.DataFrame({ col_name.split('.')[0] : np.array(dtset[pfm])}, index = dtset.index)
                    '''
                        < pandas.DataFrame ( i = 0 ) >
                            #evals  trial1_standard
                                50          f(x_50)
                                100         f(x_100)
                                :               :
                    '''
                else:
                    # add database to dataframe
                    df[pfm][col_name] = np.array(dtset[pfm])
                    '''
                        < pandas.DataFrame ( i = k  ) >
                            #evals  trial1_standard     ...     trialk_standard
                                50          f(x_50)     ...             f(x_50)
                                100         f(x_100)    ...             f(x_100)
                                :               :       ...                 :
                    '''
        # Summarize all trials
        for pfm in self.performances:
            self.path_trials[pfm] = os.path.join(self.path_out, self.cnf.filename['result-all'](self.fnc.prob_name,pfm))
            Stdio.writeDatabase(df[pfm], self.path_trials[pfm], index=True)
            log(self, f'Output {pfm} trials aggregation ({self.path_trials[pfm].split(os.path.sep)[-1]})')


    def outStandardProcessing(self) -> None:
        '''Make statistics process and best fitness curve from the standard database
        '''
        path_out = Stdio.makeDirectory(self.path_out_parent, self.cnf.dirname['standard'], confirm=False)
        for pfm in self.performances:
            # Read aggregation file
            df = Stdio.readDatabase(self.path_trials[pfm])
            if not df is None:
                # Statistics process
                # Five-number summary (min, median, max, quantile-value)
                # mean, standard deviation
                _min, _max, _q25, _med, _q75, _ave, _std = [], [], [], [], [], [], []
                for i in range(len(df.index)):
                    dtset   = np.array(df.loc[df.index[i]])
                    res     = np.percentile(dtset, [25, 50, 75])
                    _min.append(dtset.min())
                    _max.append(dtset.max())
                    _q25.append(res[0])
                    _med.append(res[1])
                    _q75.append(res[2])
                    _ave.append(dtset.mean())
                    _std.append(np.std(dtset))

                # make dataframe
                df_std = pd.DataFrame({
                    'min' : np.array(_min),
                    'q25' : np.array(_q25),
                    'med' : np.array(_med),
                    'q75' : np.array(_q75),
                    'max' : np.array(_max),
                    'ave' : np.array(_ave),
                    'std' : np.array(_std)
                    },index = df.index)

                # output database
                path_fit_tbl = os.path.join(path_out,self.cnf.filename['result-stat'](self.prob_name,pfm))
                Stdio.writeDatabase(df_std, path_fit_tbl, index=True)
                log(self,f'Output {pfm} curve ({path_fit_tbl.split(os.path.sep)[-1]})')

                # make best fitness curve (graph)
                title = f'{pfm} Curve ({self.fnc.prob_name})'
                xlabel, ylabel = 'FEs', pfm
                path_fit_fig = os.path.join(path_out,self.cnf.filename['result-stat-image'](self.prob_name,pfm))
                yscale = 'log' if ( df_std['q25'].min()!=0 and (df_std['q75'].max()/df_std['q25'].min()) > 10**2 )  else 'linear'
                Stdio.drawFigure(
                    x=df_std.index,
                    y=df_std['med'],
                    y_q25=df_std['q25'],
                    y_q75=df_std['q75'],
                    title=title,
                    xlim=(0,self.cnf.max_evals),
                    # ylim=0,
                    yscale=yscale,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    grid=True,
                    path_out=path_fit_fig
                )
                log(self,f'Output {pfm} curve ({path_fit_fig.split(os.path.sep)[-1]})')
            else:
                log(self, f'Error: Cannot read database std ({self.path_trials}).', output=sys.stderr)


    def outAdditionalProcessing(self) -> None:
        '''Output Additional Grouping
        '''
        # Population
        if self.cnf.log['population']['out']:
            path_out = Stdio.makeDirectory(self.path_out_parent, self.cnf.dirname['population'], confirm=False)
            path_pop = os.path.join(self.path_dt, self.cnf.filename['result-pop'](self.trial))
            max_trial = Stdio.getNumberOfFiles(self.path_dt, self.cnf.filename['result-pop']('*'))
            df = Stdio.readDatabase(path_pop).dropna()
            if not df is None:
                n_trials = len(df.index.drop_duplicates())
                # Visualization (first-only, only self.trial)
                if self.cnf.log['population']['visual']:
                    visual_type = 'tSNE'
                    position_columns = df.columns[df.columns.str.startswith('x')]
                    # t-SNE
                    perplexities = (2, 5, 30, 50, 100)
                    divs = tuple(df['div'].drop_duplicates())
                    for perplexity in perplexities:
                        for div in divs:
                            path_visual = os.path.join(path_out, self.cnf.filename['visual'](self.trial, visual_type,f'{self.prob_name}_p={perplexity}_div={div+1}'))
                            title = f'{visual_type} 2-D Position  div={div+1} p={perplexity} ({self.fnc.prob_name})'
                            xlabel, ylabel = 'component-1', 'component-2'
                            option = 'alpha=0.5'
                            df_div = df.query('div==0')[position_columns].dropna(how='all', axis=1)
                            x_2d = TSNE(n_components=2, perplexity=perplexity, learning_rate='auto', init='random').fit_transform(df_div)
                            assert len(df_div.columns)==self.opts[self.trial].dim[div] , f'Error: index of the dataframe ({path_pop}) is invalid.'
                            Stdio.drawFigure(
                                x=x_2d[:,0].reshape(-1, n_trials),
                                y=x_2d[:,1].reshape(-1, n_trials),
                                figsize=(6,4),
                                title=title,
                                xlabel=xlabel,
                                ylabel=ylabel,
                                cmap=self.cmap_cont,
                                cmap_lim=(df.index.min(), df.index.max()),
                                option=option,
                                colorbar=True,
                                draw_type='scatter',
                                path_out=path_visual
                            )
                            log(self,f'Output compress position by {visual_type} ({os.path.basename(path_visual)})')
            else:
                log(self, f'Error: Cannot read database pop ({path_pop}).', output=sys.stderr)

            # Best fitness and diversity curve (all trials)
            if self.cnf.log['population']['diversity']:
                path_fitdiv = os.path.join(path_out, self.cnf.filename['fit-div'](self.trial, self.prob_name))
                path_fitdiv_img = os.path.join(path_out, self.cnf.filename['fit-div-image'](self.trial, self.prob_name))
                title = f'Best Fitness and Diversity Curve ({self.fnc.prob_name})'
                xlabel, ylabel, ylabel2 = 'FEs', 'Fitness value', 'Diversity Measurement'
                div_column_name = 'Diversity'
                df_fitdiv = None
                for i in range(self.cnf.initial_seed, max_trial+self.cnf.initial_seed):
                    filename = self.cnf.filename['result-pop'](i)
                    path_div = os.path.join(self.path_dt, filename)
                    df_dvs = Stdio.readDatabase(path_div)
                    if not df_dvs is None:
                        vals = df_dvs[div_column_name].groupby('FEs').mean()
                        if i == 1:
                            df_fitdiv = pd.DataFrame({ filename : np.array(vals)}, index = df_dvs.index.drop_duplicates())
                        else:
                            df_fitdiv[filename] = np.array(vals)
                    else:
                        log(self, f'Error: Cannot read database fitdiv({path_div}).', output=sys.stderr)
                        break

                if not df_fitdiv is None:
                    # Statistics process
                    # Five-number summary (min, median, max, quantile-value)
                    # mean, standard deviation
                    _min, _max, _q25, _med, _q75, _ave, _std = [], [], [], [], [], [], []
                    for i in range(len(df_fitdiv.index)):
                        _df   = np.array(df_fitdiv.loc[df_fitdiv.index[i]])
                        res   = np.percentile(_df, [25, 50, 75])
                        _min.append(_df.min())
                        _max.append(_df.max())
                        _q25.append(res[0])
                        _med.append(res[1])
                        _q75.append(res[2])
                        _ave.append(_df.mean())
                        _std.append(np.std(_df))
                    # make dataframe
                    df_dvs = pd.DataFrame({
                        'dvs_min' : np.array(_min),
                        'dvs_q25' : np.array(_q25),
                        'dvs_med' : np.array(_med),
                        'dvs_q75' : np.array(_q75),
                        'dvs_max' : np.array(_max),
                        'dvs_ave' : np.array(_ave),
                        'dvs_std' : np.array(_std)
                        },index = df_fitdiv.index)
                    path_fit_tbl = os.path.join(self.path_out_parent, self.cnf.dirname['standard'],self.cnf.filename['result-stat'](self.prob_name,'BestFitness'))
                    df_std = Stdio.readDatabase(path_fit_tbl)
                    df_std.set_axis([ f'bf_{col}' for col in df_std.columns], axis='columns')
                    df_fitdiv = pd.merge(df_std, df_dvs, how='outer', left_index=True, right_index=True)
                    df_fitdiv
                    Stdio.writeDatabase(df_fitdiv, path_fitdiv, index=True)
                    Stdio.drawFigure(
                        x=df_std.index,
                        y=df_std['med'],
                        y_q25=df_std['q25'],
                        y_q75=df_std['q75'],
                        x2=df_dvs.index,
                        y2=df_dvs['dvs_med'],
                        y2_q25=df_dvs['dvs_q25'],
                        y2_q75=df_dvs['dvs_q75'],
                        title=title,
                        label=ylabel,
                        label2=ylabel2,
                        grid=True,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        ylabel2=ylabel2,
                        xlim=(df_std.index.min(),df_std.index.max()),
                        yscale='log',
                        color='orange',
                        color2='green',
                        path_out=path_fitdiv_img,
                        legend=True
                    )
                    log(self, f'Output best fitness and diversity curve ({path_fitdiv.split(os.path.sep)[-1]})', output=sys.stderr)
                else:
                    log(self, f'Error: Cannot read database pop ({path_pop}).', output=sys.stderr)

        # Grouping
        if self.cnf.log['grouping']['out']:
            path_out = Stdio.makeDirectory(self.path_out_parent, self.cnf.dirname['grouping'], confirm=False)

            filenames = set([ opt.group_path for opt in self.opts ])

            for filename in filenames:
                filename_group = os.path.splitext(os.path.basename(filename))[0]
                path_group = os.path.join(path_out, self.cnf.filename['grouping'](filename_group))
                df_group = Stdio.readDatabase(filename)
                if df_group is None:
                    log(self, f'Error: Cannot read group database ({filename}).', output=sys.stderr)
                    break
                df_group = df_group.iloc[:,df_group.columns.str.contains('sep')].dropna(how='all')
                max_nodes_per_div = len(df_group)
                G = nx.Graph()
                pos = {}
                x_scale, y_scale = 2, 1.3
                color = {
                    'seps_node'     : (11/255,120/255,190/255), # TF-like-blue
                    'nonseps_node'  : (245/255,147/255,34/255), # TF-like-orange
                    'link'          : (255/255,99/255,71/255),  # tomato
                }
                node_size = 150
                edges_weight_off, edges_weight_on = 0, 1
                nodes_color = {}
                edges_width = []

                for x_pos, group_name in enumerate(df_group.columns):
                    node_names = df_group.loc[:,group_name].dropna().astype(int).values
                    nodes_per_div = len(node_names)
                    offset = (max_nodes_per_div - nodes_per_div)/2
                    # Generate nodes
                    G.add_nodes_from(node_names)
                    group_color = color['nonseps_node'] if 'nonsep' in group_name else color['seps_node']
                    edges_set = []
                    for move,node_name in enumerate(node_names):
                        # Generate position
                        pos[node_name] = np.array([x_scale*x_pos,offset+y_scale*move])
                        # Set color
                        nodes_color[node_name] = group_color
                        # Generate Edge
                        if node_name != node_names[-1]:
                            weight = edges_weight_on if 'nonsep' in group_name else edges_weight_off
                            edges_set.append((node_names[move], node_names[move+1], weight))
                            edges_width.append(weight)
                        else:
                            G.add_weighted_edges_from(edges_set,weight='weight')

                # Plot figure
                plt.figure(figsize=(64,36))
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=list(nodes_color.values()), alpha=0.8)
                edges_color = [(color['link'][0], color['link'][1], color['link'][2], edges_weight_on)] * len(pos)
                edges = nx.draw_networkx_edges(G, pos, edge_color=edges_color, width=edges_width)
                labels = nx.draw_networkx_labels(G, pos, font_size=6)
                fig,ax = plt.gcf(),plt.gca()
                ax.set_axis_off()
                fig.savefig(path_group, dpi=100)
                log(self, f'Output group file ({path_group.split(os.path.sep)[-1]})')


    @staticmethod
    def analyzeTrials(path_result_fnc:str) -> None:
        # Configuration
        path_trials = os.path.join(path_result_fnc, 'trials')
        path_parent = os.path.dirname(path_result_fnc)
        prob_name = os.path.basename(path_result_fnc)
        filename_trial = 'trial*_std.csv'
        filename_summary = f'all_trials_{prob_name}.xlsx'
        filename_fit_tbl = f'stat-fitness_{prob_name}.xlsx'
        filename_fit_fig = f'stat-fitness_{prob_name}.png'


        # [1] Aggregate Trials
        # output: all_trials_fnc.csv (overwrite)
        df = None
        max_trial = Stdio.getNumberOfFiles(path_trials, filename_trial)
        for i in range(1, max_trial+1):
            filename = f'trial{i}_std'
            path_dt = os.path.join(path_trials, f'{filename}.csv')
            dtset = Stdio.readDatabase(path_dt)
            # make database
            if i == 1:
                df = pd.DataFrame({ filename : np.array(dtset['Fitness'])}, index = dtset.index)
            else:
                df[filename] = np.array(dtset['Fitness'])
        path_summary = os.path.join(path_result_fnc, filename_summary)
        Stdio.writeDatabase(df, path_summary, index=True)
        log('DataProcessing', f'Output trials aggregation ({os.path.basename(path_summary)})')

        # [2] Output fitness Curve
        # output: stat_fitness_fnc.csv / stat_fitness_fnc.png
        path_out = Stdio.makeDirectory(path_parent, '_std', confirm=False)
        # statistics process
        # min, max, quantile-value, mean, standard deviation
        _min, _max, _q25, _med, _q75, _ave, _std = [], [], [], [], [], [], []
        for i in range(len(df.index)):
            dtset   = np.array(df.loc[df.index[i]])
            res     = np.percentile(dtset, [25, 50, 75])
            _min.append(dtset.min())
            _max.append(dtset.max())
            _q25.append(res[0])
            _med.append(res[1])
            _q75.append(res[2])
            _ave.append(dtset.mean())
            _std.append(np.std(dtset))
        # make dataframe
        df_std = pd.DataFrame({
            'min' : np.array(_min),
            'q25' : np.array(_q25),
            'med' : np.array(_med),
            'q75' : np.array(_q75),
            'max' : np.array(_max),
            'ave' : np.array(_ave),
            'std' : np.array(_std)
            },index = df.index)
        # output database
        path_fit_tbl = os.path.join(path_out,filename_fit_tbl)
        Stdio.writeDatabase(df_std, path_fit_tbl, index=True)
        log('DataProcessing', f'Output fitness curve ({os.path.basename(path_fit_tbl)})')
        # make best fitness curve (graph)
        title = 'Fitness Curve ({})'.format(prob_name)
        xlabel, ylabel = 'Evaluations', 'Fitness value'
        path_fit_fig = os.path.join(path_out,filename_fit_fig)
        yscale = 'log' if ( df_std['q25'].min()!=0 and (df_std['q75'].max()/df_std['q25'].min()) > 10**2 )  else 'linear'
        Stdio.drawFigure(
            x=df_std.index,
            y=df_std['med'],
            y_q25=df_std['q25'],
            y_q75=df_std['q75'],
            title=title,
            xlim=(0,df.index.max()),
            yscale=yscale,
            xlabel=xlabel,
            ylabel=ylabel,
            grid=True,
            path_out=path_fit_fig
        )
        log('DataProcessing', f'Output fitness curve ({os.path.basename(path_fit_fig)})')


if __name__ == "__main__":
    #----------------------
    # Summarize all trials
    #----------------------

    # Configuration
    path_result_fnc = r"C:\Users\taichi\Documents\SourceCode\AQFPLogic-Optimization\_result\result_ver2.2_test_100k-evals_11-trial_CCDO-EG4_2021-05-18_06-24-14\test"
    DataProcessing.analyzeTrials(path_result_fnc)
