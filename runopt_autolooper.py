# Run Auto Looper
# version 1.6 (2021/06/25)

import subprocess   as sp
import os
import sys
from tqdm import trange

def runAutoLooper():
    '''
        設定変更自動ループ処理
    '''
    path_in         = 'config.py'            # Configurationファイル
    file_encoding   = 'UTF-8'                # 文字コード
    contents = {
        'print' : 'Abbreviation Solution Expression Optimization :',
        'lines' : {
            'line1' : 60
            # 'line2' : 33
        },
        'params' : {
            'param1' : ['True', 'False']
            # 'param2' : ['CCDGA', 'DGA']
        }
    }
    replace_num = 1
    pyversion = '3.9'

    try:
        current_dir = os.getcwd()
        if os.path.split(current_dir)[1] != 'optimizer':
            os.chdir(os.path.join('.','optimizer'))
        ''' Module Installer '''
        modules = ['numpy', 'matplotlib', 'pandas', 'xlrd', 'openpyxl', 'jinja2', 'networkx', 'pandas-profiling']
        for module in modules:
            result = sp.run(('py',f'-{pyversion}', '-m', 'pip','install', module), shell=True)

        for i in trange(len(contents['params']['param1']), desc=contents['print'].replace(':','')):
            # 初回以外パラメータ変更
            if not i == 0 :
                # read
                with open(path_in, mode='r', encoding=file_encoding) as f:
                    body = f.readlines()
                # change
                for line, param in zip(contents['lines'].values(), contents['params'].values()):
                    body[line-1] = body[line-1].replace(param[i-1],param[i],replace_num)
                # write
                with open(path_in, mode='w', encoding=file_encoding) as f:
                    f.writelines(body)

            logA = '＊'*5 + ' < ' + contents['print']
            for param in contents['params'].values():
                logA += f' {param[i]}'
            logB = ' > ' + '＊'*5
            print(logA+logB)

            # result = sp.Popen(('py', f'-{pyversion}', 'runopt.py'), shell=True)
            # result.wait()

    except TypeError:
        print('Error: ', file=sys.stderr)
    except Exception as e:
        print('Error: '+ e, file=sys.stderr)


''' main '''
if __name__ == '__main__':
    runAutoLooper()