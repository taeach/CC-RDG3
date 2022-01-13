# Data Processing
# version 1.1 (2022/01/13)

import os
import sys
from typing             import Any
import numpy            as np
import pandas           as pd
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
                self.df_std = pd.DataFrame({
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
                Stdio.writeDatabase(self.df_std, path_fit_tbl, index=True)
                log(self,f'Output {pfm} curve ({path_fit_tbl.split(os.path.sep)[-1]})')

                # make best fitness curve (graph)
                title = f'{pfm} Curve ({self.fnc.prob_name})'
                xlabel, ylabel = 'FEs', pfm
                path_fit_fig = os.path.join(path_out,self.cnf.filename['result-stat-image'](self.prob_name,pfm))
                yscale = 'log' if ( self.df_std['q25'].min()!=0 and (self.df_std['q75'].max()/self.df_std['q25'].min()) > 10**2 )  else 'linear'
                Stdio.drawFigure(
                    x=self.df_std.index,
                    y=self.df_std['med'],
                    y_q25=self.df_std['q25'],
                    y_q75=self.df_std['q75'],
                    title=title,
                    xlim=(0,self.cnf.max_evals),
                    ylim=0,
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
            df = Stdio.readDatabase(path_pop)
            if not df is None:
                num_k = len(df['k'].drop_duplicates())
                # Visualization (first-only, only self.trial)
                if self.cnf.log['population']['visual']:
                    visual_type = 'tSNE'
                    position_columns = df.columns[df.columns.str.startswith('x')]
                    # t-SNE
                    perplexities = [2, 5, 30, 50, 100]
                    for perplexity in perplexities:
                        path_visual = os.path.join(path_out, self.cnf.filename['visual'](visual_type,self.prob_name,f'p={perplexity}'))
                        title = f'{visual_type} 2-D Position p={perplexity} ({self.fnc.prob_name})'
                        xlabel, ylabel = 'component-1', 'component-2'
                        option = 'alpha=0.5'
                        x_2d = TSNE(n_components=2, perplexity=perplexity).fit_transform(df[position_columns])
                        assert len(df)%num_k == 0 , f'Error: index of the dataframe ({path_pop}) is invalid.'
                        Stdio.drawFigure(
                            x=x_2d[:,0].reshape(-1, num_k),
                            y=x_2d[:,1].reshape(-1, num_k),
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
                        log(self,f'Output compress position by {visual_type} ({path_visual.split(os.path.sep)[-1]})')
            else:
                log(self, f'Error: Cannot read database pop ({path_pop}).', output=sys.stderr)

            # Best fitness and diversity curve (all trials)
            if self.cnf.log['population']['diversity']:
                path_fitdiv = os.path.join(path_out, self.cnf.filename['fit-div'](self.trial, self.prob_name))
                title = f'Best Fitness and Diversity Curve ({self.fnc.prob_name})'
                xlabel, ylabel, ylabel2 = 'FEs', 'Fitness value', 'Diversity Measurement'
                div_column_name = 'Diversity'
                df_fitdiv = None
                for i in range(self.cnf.initial_seed, max_trial+self.cnf.initial_seed):
                    filename = self.cnf.filename['result-pop'](i)
                    path_div = os.path.join(self.path_dt, filename)
                    df_div = Stdio.readDatabase(path_div)
                    if not df_div is None:
                        if i == 1:
                            df_fitdiv = pd.DataFrame({ filename : np.array(df_div[div_column_name])}, index = df_div.index)
                        else:
                            df_fitdiv[filename] = np.array(df_div[div_column_name])
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
                    df_div = pd.DataFrame({
                        'min' : np.array(_min),
                        'q25' : np.array(_q25),
                        'med' : np.array(_med),
                        'q75' : np.array(_q75),
                        'max' : np.array(_max),
                        'ave' : np.array(_ave),
                        'std' : np.array(_std)
                        },index = df_fitdiv.index)

                    Stdio.drawFigure(
                        x=self.df_std.index,
                        y=self.df_std['med'],
                        y2=df_div['max'],
                        title=title,
                        label=ylabel,
                        label2=ylabel2,
                        grid=True,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        ylabel2=ylabel2,
                        xlim=(self.df_std.index.min(),self.df_std.index.max()),
                        yscale='log',
                        color='orange',
                        color2='green',
                        path_out=path_fitdiv,
                        legend=True
                    )
                    log(self, f'Output best fitness and diversity curve ({path_fitdiv.split(os.path.sep)[-1]})', output=sys.stderr)
                else:
                    log(self, f'Error: Cannot read database pop ({path_pop}).', output=sys.stderr)

        # Grouping
        if self.cnf.log['grouping']['out']:
            path_out = Stdio.makeDirectory(self.path_out_parent, self.cnf.dirname['grouping'], confirm=False)
            for opt in self.opts:
                filename_group = os.path.splitext(os.path.basename(opt.group_path))[0]
                path_group = os.path.join(path_out, self.cnf.filename['grouping'](filename_group))
                df_group = Stdio.readDatabase(opt.group_path)
                if df_group is None:
                    log(self, f'Error: Cannot read database mod ({opt.group_path}).', output=sys.stderr)
                    break
                max_nodes_per_div = len(df_group.columns[df_group.columns.str.contains('org_id')])
                G = nx.Graph()
                pos = {}
                x_scale, y_scale = 4, 1
                color = {
                    'node'          : (11/255,120/255,190/255), # TF-like-blue
                    'error_node'    : (255/255,99/255,71/255),  # tomato
                    'edge'          : (245/255,147/255,34/255)  # TF-like-orange
                }
                node_size = 150
                error_nodes = [] # sample

                indices = set(df_group.index)
                # To avoid complicating, last relationship don't plot
                indices.remove(max(indices))

                for j in indices:
                    df_div = df_group.loc[j].dropna(axis=1)
                    # org_id
                    ds_div_org = df_div.loc[:,df_div.columns.str.contains('org_id')].iloc[0].astype(int)
                    nodes_per_div_org = len(ds_div_org)
                    if j==0:
                        offset = (max_nodes_per_div - nodes_per_div_org)/2
                        G.add_nodes_from(ds_div_org)
                        for _j,_org in enumerate(ds_div_org):
                            pos[_org] = np.array([x_scale*j,offset+y_scale*_j])

                    # pred_id
                    ds_div_pred = df_div['pred_id']
                    G.add_nodes_from(ds_div_pred)
                    nodes_per_div_pred = len(ds_div_pred)
                    for _j,_pred in enumerate(ds_div_pred):
                        offset = (max_nodes_per_div - nodes_per_div_pred)/2
                        pos[_pred] = np.array([x_scale*(j+1),offset+y_scale*_j])

                    # generate edge
                    # normalize fitness importance
                    max_index = df_div.loc[:,df_div.columns.str.contains('FI')].max(axis=1)
                    df_div_fi = df_div.loc[:,df_div.columns.str.contains('FI')].div(max_index, axis='index')
                    edges_set = []
                    for _col,_pred in enumerate(ds_div_pred):
                        for _ind,_org in enumerate(ds_div_org):
                            edges_set.append((_org, _pred, df_div_fi.iloc[_col,_ind]))
                    G.add_weighted_edges_from(edges_set,weight='weight')

                # Plot figure
                # x_y_length = np.array(list(pos.values())).max(axis=0)
                # fig_scale = 0.4
                # figsize = tuple(fig_scale*x_y_length)
                plt.figure(figsize=(60,30))
                nodes_color = []
                for j in G.nodes():
                    nodes_color.append(color['error_node'] if j in error_nodes else color['node'])
                nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=nodes_color, alpha=0.8)
                edges_weight = np.array(list(nx.get_edge_attributes(G, 'weight').values()))
                edges_color = []
                for edge_weight in edges_weight:
                    edges_color.append((color['edge'][0], color['edge'][1], color['edge'][2], edge_weight))
                edges = nx.draw_networkx_edges(G, pos, edge_color=edges_color, width=edges_weight)
                labels = nx.draw_networkx_labels(G, pos, font_size=6)
                fig,ax = plt.gcf(),plt.gca()
                ax.set_axis_off()
                fig.savefig(path_group, dpi=100)
                log(self, f'Output fitness importance network model-{i} ({path_group.split(os.path.sep)[-1]})', output=sys.stderr)


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
