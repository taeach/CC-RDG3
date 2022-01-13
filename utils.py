# Utilities
# version 1.2 (2022/01/13)

import  os
import  sys
import  csv
from    io          import TextIOWrapper
import  platform    as pf
import  traceback   as tb
import  shutil      as st
import  glob        as gl
from typing import Callable
import  numpy       as np
import  pandas      as pd
from matplotlib     import pyplot   as plt
from matplotlib     import colors   as cls


DEFAULT:str     = '\033[39m'
BLACK:str       = '\033[30m'
RED:str         = '\033[31m'
GREEN:str       = '\033[32m'
YELLOW:str      = '\033[33m'
BLUE:str        = '\033[34m'
MAGENTA:str     = '\033[35m'
CYAN:str        = '\033[36m'
WHITE:str       = '\033[37m'
BOLD:str        = '\033[1m'
UNDERLINE:str   = '\033[4m'
RESET:str       = '\033[0m'
COLOR_DICT:dict = {
    'Configuration' : YELLOW,
    'Function'      : CYAN,
    'DataLogger'    : BLUE,
    'DataProcessing': GREEN,
    'Optimizer'     : RED,
    'OptimizerCore' : MAGENTA
}
ENCODING:str    = 'UTF-8'


def log(attrib_name:str|Callable, message:str, output:TextIOWrapper=sys.stdout, color:str='color') -> None:
    '''Logging (instead of "print()")
    Args:
        attrib_name (str): Attribute name (e.g., class name, function name).
        message (str): Message for all positions.
        output (TextIOWrapper, optional): Output type. Defaults to sys.stdout.
        color (str, optional): Color print ('color' or COLOR_NAME or 'mono'). Defaults to 'color'.
    '''
    if not isinstance(attrib_name,str):
        attrib_name = attrib_name.__class__.__name__
    if color!='mono':
        if color=='color':
            COLOR = COLOR_DICT[attrib_name] if attrib_name in COLOR_DICT.keys() else BOLD+BLACK
        else:
            COLOR = color
        print(rf'{COLOR}[{attrib_name}]{RESET} {message}', file=output)
    else:
        print(f'[{attrib_name}] {message}', file=output)


class Stdio:
    '''Standard Input/Output library
    '''
    @staticmethod
    def moveWorkingDirectory():
        wd_path = os.getcwd()
        wd_name = os.path.split(wd_path)[1]
        # move parent
        if wd_name in ('_env', '_result'):
            os.chdir('..')
        # move child
        elif wd_name in ():
            pass

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
                print('<< output folder conflict >>')
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
            path_write = f'{path_write}.{csv_or_xlsx}'
        if csv_or_xlsx == 'xlsx':
            df.to_excel(path_write, engine='openpyxl', index=index, header=header)
        elif csv_or_xlsx == 'csv':
            df.to_csv(path_write, index=index, header=header)

    @staticmethod
    def readDatabase(path_read:str, index_col:int|None=0, header:int|None=0, names:tuple|list|None=None) -> pd.DataFrame:
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
    def saveExperimentalData(path_out:str, data_dict:dict, file_name:str|None=None, seed:int=0, csv_or_xlsx:str='csv', display='vertical'):
        '''save experimental data
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
    def drawFigure(
        x:list[int,float,str],
        y:list[int,float]|np.ndarray|pd.core.series.Series|None=None,
        y2:list[int,float]|np.ndarray|pd.core.series.Series|None=None,
        y_q25:list[int,float]|np.ndarray|pd.core.series.Series|None=None,
        y_q75:list[int,float]|np.ndarray|pd.core.series.Series|None=None,
        figsize:tuple[int,float]=(10,4),
        title:str|None=None,
        label:str|list[str]='line-1',
        label2:str|list[str]='line-2',
        xlim:tuple[int,float]|list[int,float]|np.ndarray=None,
        ylim:tuple[int,float]|list[int,float]|np.ndarray=None,
        ylim2:tuple[int,float]|list[int,float]|np.ndarray=None,
        xlabel:str|None=None,
        ylabel:str|None=None,
        ylabel2:str|None=None,
        xscale:str='linear',
        yscale:str='linear',
        grid:bool=False,
        grid2:bool=False,
        legend:bool=False,
        linestyle:str|list[str]='-',
        color:str|list[str]='blue',
        color2:str|list[str]='orange',
        cmap:cls.Colormap|None=None,
        cmap_lim:tuple|list|None=None,
        cmap2:cls.Colormap|None=None,
        cmap2_lim:tuple|list|None=None,
        option:str='',
        option2:str='',
        colorbar:bool=False,
        draw_type:str='plot',
        show:bool=False,
        save:bool=True,
        path_out:str|None=None,
        dpi:int=300,
        output:bool=False):
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

