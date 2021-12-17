# Data Processing
# version 16.4 (2021/12/1)

import numpy                        as np
import pandas                       as pd
import os
import sys
import matplotlib
from config  import Configuration
cf_for_ml = Configuration()
if not cf_for_ml.anime_out:
    matplotlib.use('Agg')
from matplotlib import pyplot       as plt
from matplotlib import animation    as ani
from sklearn import manifold        as mf
import networkx as nx
from logger import Stdio
''' For Function Annotations '''
import config           as cf
import function         as fc

class Statistics:
    def __init__(self, cnf:cf.Configuration, fnc:fc.Function, path_out:str, path_dt:str):
        self.path_out   = path_out
        self.path_out_parent   = os.path.sep.join(self.path_out.split(os.path.sep)[:-1])
        self.path_dt    = path_dt
        self.cnf        = cnf
        self.fnc        = fnc
        self.prob_name  = self.fnc.prob_name
        self.cmap_10    = plt.get_cmap('tab10')
        self.cmap_cont  = plt.get_cmap('RdYlBu')
        self.trial      = 1
        plt.rcParams['font.family'] = 'Times New Roman' # Fonts
        self.performances = ['CalcTime[ms]', 'FitnessValue', 'OrgFitnessValue','TotalWL', 'NoVW', 'NoVL', 'LSCount']

    def outStatistics(self) -> None:
        '''
            make statistics process and visualization from the standard and detail database
        '''
        if self.cnf.log['standard']['out']:
            self.aggregateTrials()
            self.outStatisticsStandard()
            self.outStatisticsDetail()

    def aggregateTrials(self) -> None:
        '''
            make aggregation file from the standard database
        '''
        # read csv file
        max_trial = Stdio.getNumberOfFiles(self.path_dt,self.cnf.filename['result']('*'))

        df = {}
        self.path_trials = {}
        for i in range(self.cnf.initial_seed, max_trial+self.cnf.initial_seed):
            # extract database from trialN_standard.csv
            filename = self.cnf.filename['result'](i)
            path_dt = os.path.join(self.path_dt, filename)
            dtset = Stdio.readDatabase(path_dt)
            for pfm in self.performances:
                col_name = filename.split('.')[0]
                # make database
                if i == 1:
                    # make dataframe
                    df[pfm] = pd.DataFrame({ col_name.split('.')[0] : np.array(dtset[pfm])}, index = dtset.index)
                    '''
                        < pandas.DataFrame ( i = 0 ) >
                            #evals  trial1_standard
                                50          f(x_50)
                                100          f(x_100)
                                :               :
                    '''
                else:
                    # add database to dataframe
                    df[pfm][col_name] = np.array(dtset[pfm])
                    '''
                        < pandas.DataFrame ( i = k  ) >
                            #evals  trial1_standard     ...     trialk_standard
                                50          f(x_50)     ...             f(x_50)
                                100          f(x_100)    ...             f(x_100)
                                :               :       ...                 :
                    '''

        for pfm in self.performances:
            self.path_trials[pfm] = os.path.join(self.path_out, self.cnf.filename['result-all'](self.fnc.prob_name,pfm))
            Stdio.writeDatabase(df[pfm], self.path_trials[pfm], index=True)
            print('[DataProcessing] Output {} trials aggregation ({})'.format(pfm, self.path_trials[pfm].split(os.path.sep)[-1]))


    def outStatisticsStandard(self) -> None:
        '''
            make statistics process and best fitness curve from the standard database
        '''
        path_out = Stdio.makeDirectory(self.path_out_parent, '_std', confirm=False)
        for pfm in self.performances:
            # read csv file
            df = Stdio.readDatabase(self.path_trials[pfm])
            if not df is None:
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
                print('[DataProcessing] Output {} curve ({})'.format(pfm,path_fit_tbl.split(os.path.sep)[-1]))

                # make best fitness curve (graph)
                title = f'{pfm} Curve ({self.fnc.prob_name})'
                xlabel, ylabel = 'Evaluations', pfm
                path_fit_fig = os.path.join(path_out,self.cnf.filename['result-stat-image'](self.prob_name,pfm))
                yscale = 'log' if ( self.df_std['q25'].min()!=0 and (self.df_std['q75'].max()/self.df_std['q25'].min()) > 10**2 )  else 'linear'
                if pfm == 'NoVL':
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
                else:
                    Stdio.drawFigure(
                        x=self.df_std.index,
                        y=self.df_std['med'],
                        y_q25=self.df_std['q25'],
                        y_q75=self.df_std['q75'],
                        title=title,
                        xlim=(0,self.cnf.max_evals),
                        yscale=yscale,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        grid=True,
                        path_out=path_fit_fig
                    )
                print('[DataProcessing] Output {} curve ({})'.format(pfm, path_fit_fig.split(os.path.sep)[-1]))
            else:
                print('[DataProcessing] Error: Cannot read database std ({}).'.format(self.path_trials), file=sys.stderr)


    def outStatisticsDetail(self) -> None:
        # input protocol
        filename_pop = 'trial*_pop.csv'
        filename_oth = 'trial*_others.csv'
        filename_mod = 'trial*_model-$.csv'
        if self.cnf.log['population']['out']:
            # output path
            path_out = Stdio.makeDirectory(self.path_out_parent, '_pop', confirm=False)
            # input path
            path_pop = os.path.join(self.path_dt, filename_pop.replace('*', str(self.trial)))
            path_oth = os.path.join(self.path_dt, filename_oth.replace('*', str(self.trial)))
            # get trials
            max_trial = Stdio.getNumberOfFiles(self.path_dt, filename_oth)
            # read csv
            df = Stdio.readDatabase(path_pop)
            if not df is None:
                num_k = len(df['k'].drop_duplicates())

                # visualization (only self.trial)
                if self.cnf.log['population']['out']:
                    visual_type = 't-SNE'
                    # output path
                    path_visual = os.path.join(path_out,'trial{}_pos_{}_{}.png'.format(self.trial, visual_type, self.prob_name))
                    position_columns = df.columns[df.columns.str.startswith('xk')]
                    # t-SNE
                    if visual_type == 't-SNE':
                        perplexities = [2, 5, 30, 50, 100]
                        for perplexity in perplexities:
                            path_visual = os.path.join(path_out,'trial{}_pos_{}_{}_p={}.png'.format(self.trial, visual_type, self.prob_name, perplexity))
                            title = '{} 2-d position p={} ({})'.format(visual_type, perplexity, self.fnc.prob_name)
                            xlabel, ylabel = 'component-1', 'component-2'
                            indices = df.index.drop_duplicates()
                            option = 'alpha=0.5'
                            x_2d = mf.TSNE(n_components=2, perplexity=perplexity).fit_transform(df[position_columns])
                            assert len(df)%num_k == 0 , 'Error: index of the dataframe ({}) is invalid.'.format(path_pop)
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
                            print('[DataProcessing] Output compress position by {} ({})'.format(visual_type, path_visual.split(os.path.sep)[-1]))
                    else:
                        pass

                # current fitness and predict curve (only self.trial)
                if self.cnf.log['population']['out']:
                    # assert self.cnf.dmod_out==True, 'Error: dchg_out requires dmod_out==True'
                    # read csv
                    df_oth = Stdio.readDatabase(path_oth)
                    if not df_oth is None:
                        # output path
                        path_fitmod = os.path.join(path_out,'trial{}_fitness-with-pred_{}.png'.format(self.trial, self.fnc.prob_name))
                        title = 'Current fitness and model changes curve ({})'.format(self.fnc.prob_name)
                        xlabel, ylabel, ylabel2 = 'Evaluations', 'Fitness value', 'Model changes'
                        fitness_column, changes_column = 'fk(x)', 'SM:f_pred'

                        fitness = [ np.array(df.loc[df['k']==k, fitness_column])  for k in range(num_k) ]
                        Stdio.drawFigure(
                            x=df.index.drop_duplicates(),
                            y=fitness,
                            y2=df_oth[changes_column],
                            title=title,
                            label=ylabel,
                            label2=ylabel2,
                            grid=True,
                            xlabel=xlabel,
                            ylabel=ylabel,
                            ylabel2=ylabel2,
                            xlim=(df.index.min(),df.index.max()),
                            yscale='log',
                            color2='slategray',
                            cmap=self.cmap_cont,
                            path_out=path_fitmod,
                            legend=True
                        )
                        print('[DataProcessing] Output fitness and predict fitness curve ({})'.format(path_fitmod.split(os.path.sep)[-1]))
                    else:
                        print('[DataProcessing] Error: Cannot read database oth ({}).'.format(path_oth), file=sys.stderr)

                # best fitness and diversity curve
                if self.cnf.log['population']['out']:
                    # assert self.cnf.dmod_out==True, 'Error: ddiv_out requires dmod_out==True'
                    # output path
                    path_fitdiv = os.path.join(path_out, 'bestfit-with-div_{}.png'.format(self.fnc.prob_name))
                    title = 'Best fitness and diversity curve ({})'.format(self.fnc.prob_name)
                    xlabel, ylabel, ylabel2 = 'Evaluations', 'Fitness value', 'Diversity'
                    fitness_column, div_column = 'fk(x)', 'D(x)'
                    df_oth = None

                    for i in range(1,max_trial+1):
                        filename = filename_oth.replace('#',str(i))
                        _path_oth = os.path.join(self.path_dt, filename)
                        # read csv
                        _df_oth = Stdio.readDatabase(_path_oth)
                        if not _df_oth is None:
                            if i == 1:
                                df_oth = pd.DataFrame({ filename : np.array(_df_oth[div_column])}, index = _df_oth.index)
                            else:
                                df_oth[filename] = np.array(_df_oth[div_column])
                        else:
                            print('[DataProcessing] Error: Cannot read database oth ({}).'.format(_path_oth), file=sys.stderr)
                            break

                    if not df_oth is None:
                        # statistics process
                        # min, max, quantile-value, mean, standard deviation
                        _min, _max, _q25, _med, _q75, _ave, _std = [], [], [], [], [], [], []
                        for i in range(len(df_oth.index)):
                            _df   = np.array(df_oth.loc[df_oth.index[i]])
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
                            },index = df_oth.index)

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
                        print('[DataProcessing] Output best fitness and diversity curve ({})'.format(path_fitdiv.split(os.path.sep)[-1]))

            else:
                print('[DataProcessing] Error: Cannot read database pop ({}).'.format(path_pop), file=sys.stderr)

        if self.cnf.log['population']['out']:
            # assert self.cnf.dmod_out==True, 'Error: dnet_out requires dmod_out==True'
            # output path
            path_out = Stdio.makeDirectory(self.path_out_parent, '_model', confirm=False)
            # read csv
            _filename_mod = filename_mod.replace('*', str(self.trial))
            max_models = Stdio.getNumberOfFiles(self.path_dt, _filename_mod, mark='$')
            for i in range(1,max_models+1):
                path_finet = os.path.join(path_out,'trial{}_fi-network_model-{}_{}.png'.format(self.trial, i, self.fnc.prob_name))
                path_mod = os.path.join(self.path_dt, _filename_mod.replace('$', str(i)))
                df_mod = Stdio.readDatabase(path_mod)
                if df_mod is None:
                    print('[DataProcessing] Error: Cannot read database mod ({}).'.format(path_mod), file=sys.stderr)
                    break
                # title = 'Fitness importance network [model-{}] ({})'.format(i, self.fnc.prob_name)
                max_nodes_per_div = len(df_mod.columns[df_mod.columns.str.contains('org_id')])
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

                indices = set(df_mod.index)
                # To avoid complicating, last relationship don't plot
                indices.remove(max(indices))

                for j in indices:
                    df_div = df_mod.loc[j].dropna(axis=1)
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

                # plot figure
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
                fig.savefig(path_finet, dpi=100)

                print('[DataProcessing] Output fitness importance network model-{} ({})'.format(i, path_finet.split(os.path.sep)[-1]))


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
        print(f'[DataProcessing] Output trials aggregation ({os.path.basename(path_summary)})')

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
        print('[DataProcessing] Output fitness curve ({})'.format(os.path.basename(path_fit_tbl)))
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
        print('[DataProcessing] Output fitness curve ({})'.format(os.path.basename(path_fit_fig)))


class Plot:
    def __init__(self, cnf:cf.Configuration, fnc:fc.Function):
        self.cnf    = cnf
        self.fnc    = fnc
        self.x      = []                            # x[gen,pop,dim]
        # function color mesh graph
        self.cmap   = 'viridis'                     # color map
        self.fnc_plot = 200                         # function plot number(200)
        self.X1_fnc,self.X2_fnc,self.F = [],[],[]   # function parameter
        # population graph
        self.graph_color = 'magenta'                # individual color
        # animation
        self.figsize=(9,9)                          # figure size
        self.ims    = []                            # frame
        self.interval_ms = 10                       # interval [ms]
        self.frames = int(self.cnf.max_evals/self.cnf.anime_interval) + 1

        # set parameters
        plt.rcParams['font.family'] ='sans-serif'   # using font-family
        plt.rcParams['xtick.direction'] = 'in'      # x-axis direction (in/out/inout)
        plt.rcParams['ytick.direction'] = 'in'      # y-axis direction (in/out/inout)


    def getDataX(self, x:np.ndarray, total_evals:int) -> list:
        '''
            get position vector between animation interval
        '''
        if self.cnf.anime_out and ( total_evals == self.cnf.max_pop or (total_evals % self.cnf.anime_interval) == 0 ) :
            # pos[pop][dim]  (list)
            pos = []
            for i in range(self.cnf.max_pop):
                pos.append(list(x[i]))
            return pos


    def compressAxis(self):
        '''
            compress axis from any-dimension to 2-dimension
        '''
        self.x = np.array(self.x)
        if not self.cnf.prob_dim == 2:
            for i in range(len(self.x)):
                for j in range(len(self.x[i])):
                    odd, even = 1,1
                    for k in range(2,len(self.x[i,j])):
                        if k % 2 == 0 :
                            # odd-dimensional axis
                            self.x[i,j,0] += self.x[i,j,k]
                            odd += 1
                        else :
                            # even-dimensional axis
                            self.x[i,j,1] += self.x[i,j,k]
                            even += 1
                    # averaging (flatten)
                    self.x[i,j,0] /= odd
                    self.x[i,j,1] /= even

    def generateData(self):
        '''
            generate database for 2D graph
        '''
        # employ only 1 axis range
        # make axis plot
        x1 = np.linspace(self.fnc.axis_range[0,0], self.fnc.axis_range[0,1], self.fnc_plot)
        x2 = np.linspace(self.fnc.axis_range[0,0], self.fnc.axis_range[0,1], self.fnc_plot)
        self.X1_fnc, self.X2_fnc = np.meshgrid(x1, x2)

        # make mesh data
        X_fnc = [self.X1_fnc, self.X2_fnc]*int(self.cnf.prob_dim/2)
        if self.cnf.prob_dim % 2 == 1 :
            X_fnc.append(self.X1_fnc)

        # make function value
        for j in range(self.fnc_plot):
            for k in range(self.fnc_plot):
                X = np.array([X_fnc[i][j][k] for i in range(self.cnf.prob_dim)])
                self.F.append(self.fnc.evaluate(X))
        self.F = np.array(self.F).reshape((self.fnc_plot,self.fnc_plot))


    def initFigure(self):
        '''
            initialize function graph
        '''
        self.compressAxis()
        self.generateData()
        # make colorbar
        cb_fnc = plt.colorbar(plt.pcolormesh(self.X1_fnc, self.X2_fnc, self.F, cmap = self.cmap),orientation='vertical')


    def updateFigure(self,frame:int):
        '''
            graph plot
        '''
        if not frame==0:
            plt.cla()
        # clear graph
        if len(self.ims) > 0 :
            del self.ims[:]
        # make function color mesh graph
        im_fnc = plt.pcolormesh(self.X1_fnc, self.X2_fnc, self.F, cmap = self.cmap)
        plt.grid()
        plt.xlim(self.fnc.axis_range[0])
        plt.ylim(self.fnc.axis_range[1])
        if self.cnf.prob_dim == 2:
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
        else :
            plt.xlabel('$x_{odd}$')
            plt.ylabel('$x_{even}$')
        plt.locator_params(axis='x',nbins=10)       # x-axis number of scale (nbin)
        plt.locator_params(axis='y',nbins=10)       # y-axis number of scale (nbin)
        # make axis
        x1 = [ self.x[frame,i,0] for i in range(self.cnf.max_pop) ]
        x2 = [ self.x[frame,i,1] for i in range(self.cnf.max_pop) ]
        # generate graph frame
        im_opt = plt.scatter(x1, x2, color=self.graph_color)
        plt.title('{}-D $x$ vector behavior for {} (gen={})'.format(self.fnc.prob_dim, self.fnc.prob_name, int(self.cnf.anime_interval * frame)))
        # add graph frame
        self.ims.append(im_fnc)
        self.ims.append(im_opt)


    def plotFigureAnimation(self):
        '''
            animation plot
        '''
        if self.cnf.anime_out :
            # initialization
            fig = plt.figure(figsize=self.figsize)
            self.initFigure()
            # make animation
            #animation = ani.ArtistAnimation(fig, self.ims, interval=self.interval_ms, repeat=False, blit=True)
            animation = ani.FuncAnimation(fig, self.updateFigure, interval=self.interval_ms, repeat=False, frames=self.frames)
            # standard output of the graph
            plt.show()
            # output graph (.gif)
            # path_out = self.cnf.path_out + '/animation_' + str(self.fnc.prob_dim) + 'D_' + self.fnc.prob_name + '.gif'
            # animation.save(path_out, writer='imagemagick')
            # close graph
            plt.close()


if __name__ == "__main__":
    #----------------------
    # Summarize all trials
    #----------------------

    # Configuration
    path_result_fnc = r"C:\Users\taichi\Documents\SourceCode\AQFPLogic-Optimization\_result\result_ver2.2_test_100k-evals_11-trial_CCDO-EG4_2021-05-18_06-24-14\test"
    Statistics.analyzeTrials(path_result_fnc)
