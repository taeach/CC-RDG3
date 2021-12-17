# Function Distributer
# version 1.0 (2021/12/17)

import math  as mt
import numpy as np

from config import Configuration


class Function:
    '''Continuous Function

    Attributes:
        cnf (Configuration): Configuration class
        prob_name (str): Problem name (Function name)
        prob_dim (int): Dimension of problem（only F1~F8）
        axis_range (list or 2-d ndarray): Axis range of problem
        evaluate (function): problem_name problem to evaluate x
        total_evals (int): Total fitness evaluation(FEs)
        xopt (1-d ndarray): global optima for 

    Note:
        Problem/Function:
        - Basic Benchmark Function (F1~F8)
            ※ variable dimension (any)
        - LSGO CEC'2013 Benchmark Function (LSGO_F1~LSGO_F15)
            ※ fixed dimension (905,1000D)
    '''

    def __init__(self, cnf:Configuration, prob_name:str, prob_dim:int):
        self.cnf            = cnf
        self.prob_name      = prob_name
        self.prob_dim       = prob_dim
        self.axis_range     = []
        self.evaluate       = None
        self.total_evals    = 0
        if 'LSGO' in self.prob_name :
            self.xopt       = None

        self.setFunction()
        self.extendDomain()
        print('\t\t[ Problem : {} ]\t\t'.format(prob_name))


    def setFunction(self) -> None:
        '''Function Setter
        Assign the function with the specified problem name to `evaluate`
        '''
        if self.prob_name == 'F1':
            self.evaluate = self.F1
            self.axis_range  = [-100,100]
        elif self.prob_name == 'F2':
            self.evaluate = self.F2
            self.axis_range  = [-50, 50]
        elif self.prob_name == 'F3':
            self.evaluate = self.F3
            self.axis_range  = [-50, 50]
        elif self.prob_name == 'F4':
            self.evaluate = self.F4
            self.axis_range  = [-50, 50]
        elif self.prob_name == 'F5':
            self.evaluate = self.F5
            self.axis_range  = [-100,100]
        elif self.prob_name == 'F6':
            self.evaluate = self.F6
            self.axis_range  = [-0.5, 0.5]
        elif self.prob_name == 'F7':
            self.evaluate = self.F7
            self.axis_range  = [-500,500]
        elif self.prob_name == 'F8':
            self.evaluate = self.F8
            self.axis_range  = [-100,100]
        elif self.prob_name == 'F9':
            self.evaluate = self.F9
            self.axis_range  = [-100,100]
        elif self.prob_name == 'LSGO_F1':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F1_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F1
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-100,100]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F2':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F2_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F2
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-5, 5]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F3':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F3_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F3
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-32, 32]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F4':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F4_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F4
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-100, 100]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F5':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F5_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F5
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-5,5]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F6':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F6_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F6
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-32,32]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F7':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F7_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F7
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-100,100]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F7M':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F7M_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                pass
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-100,100]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F8':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F8_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F8
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-100,100]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F9':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F9_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F9
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-5,5]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F10':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F10_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F10
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-32,32]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F11':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F11_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F11
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-100,100]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F12':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F12_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F12
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-100,100]
            self.prob_dim = 1000
        elif self.prob_name == 'LSGO_F13':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F13_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F13
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-100,100]
            # 905D because of overlapping
            self.prob_dim = 905
        elif self.prob_name == 'LSGO_F14':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F14_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F14
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-100,100]
            # 905D because of overlapping
            self.prob_dim = 905
        elif self.prob_name == 'LSGO_F15':
            if self.cnf.prob_env_noise == 'on' :
                self.evaluate = self.LSGO_F15_NOISE
            elif self.cnf.prob_env_noise == 'off' :
                self.evaluate = self.LSGO_F15
            else :
                print('Error: Do not exist prob_env_noise {} (class {})'.format(self.cnf.prob_env_noise,self.__class__.__name__))
            self.axis_range  = [-100,100]
            self.prob_dim = 1000
        else:
            print('Error: Do not exist Function {} (class {})'.format(self.prob_name,self.__class__.__name__))
            return None

    def doEvaluate(self, x:np.ndarray):
        '''Evaluate x

        Args:
            x (np.ndarray): solution

        Returns:
            float: fitness value
        '''
        self.total_evals += 1
        return self.evaluate(x)

    def doEvaluateLog(self, x:np.ndarray):
        '''Evaluate x for log (countless)

        Args:
            x (np.ndarray): solution

        Returns:
            float: fitness value
        '''
        return self.evaluate(x)

    def resetTotalEvals(self):
        '''Reset total FEs
        '''
        self.total_evals = 0

    def extendDomain(self):
        '''Extend domain for 1-d array

        [min,max] -> [[min_1d,max_1d],[min_2d,max_2d],...[min_nd,max_nd]]
        '''
        # when axis range is 1-d array
        if np.array(self.axis_range).shape == (2,) :
            self.axis_range = np.array([self.axis_range for i in range(self.prob_dim)])


    ''' ＊＊＊＊＊  Basic  ＊＊＊＊＊ '''
    ''' Basic Benchmark Function '''

    def F1(self, x:np.ndarray):
        '''F1 : Sphere Function
        Separability    : Fully-separable
        Function Form   : Unimodal

        Args:
            x (np.ndarray): solution

        Returns:
            float: fitness value
        '''
        ret = np.sum(x**2)
        return ret

    def F2(self, x:np.ndarray):
        '''F2 : Rosenbrock Function
        Separability    : Non-separable
        Function Form   : Unimodal

        Args:
            x (np.ndarray): solution

        Returns:
            float: fitness value
        '''
        dim = len(x)
        ret = np.sum(100*(x[0:dim-1]**2 - x[1:dim])**2 + (x[0:dim-1]-1)**2)
        return ret

    def F3(self, x:np.ndarray):
        '''F3 : Ackley Function
        Separability    : Fully-separable
        Function Form   : Unimodal

        Args:
            x (np.ndarray): solution

        Returns:
            float: fitness value
        '''
        dim = len(x)
        ret = -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/dim)) - np.exp(np.sum(np.cos(2*mt.pi*x)/dim)) + 20 + np.exp(1)
        return ret

    def F4(self, x:np.ndarray):
        '''F4 : Rastrigin Function
        Separability    : Fully-separable
        Function Form   : Multimodal

        Args:
            x (np.ndarray): solution

        Returns:
            float: fitness value
        '''
        ret = np.sum(x**2 - 10*np.cos(2*mt.pi*x) + 10)
        return ret

    def F5(self, x:np.ndarray):
        '''F5 : Griewank Function
        Separability    : Fully-separable
        Function Form   : Multimodal

        Args:
            x (np.ndarray): solution

        Returns:
            float: fitness value
        '''
        dim = len(x)
        w = [i+1 for i in range(dim)]
        ret = 1 + np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(w)))
        return ret

    def F6(self, x:np.ndarray):
        '''F6 : Weierstrass Function
        Separability    : Fully-separable
        Function Form   : Multimodal

        Args:
            x (np.ndarray): solution

        Returns:
            float: fitness value
        '''
        a, b, kmax = 0.5, 3., 20
        dim =len(x)
        ret = np.sum([a**k * np.cos(2 * mt.pi * b**k * (x+0.5)) for k in range(kmax+1)]) - dim*np.sum([a**k * np.cos(2 * mt.pi * b**k * 0.5) for k in range(kmax+1)])
        return ret

    def F7(self, x:np.ndarray):
        '''F7 : Schwefel Function
        Separability    : Fully-separable
        Function Form   : Multimodal

        Args:
            x (np.ndarray): solution

        Returns:
            float: fitness value
        '''
        dim = len(x)
        ret = 418.9829 * dim - np.sum(x*np.sin(np.sqrt(abs(x))))
        return ret

    def F8(self, x:np.ndarray):
        '''F8 : Elliptic Function
        Separability    : Fully-separable
        Function Form   : Multimodal

        Args:
            x (np.ndarray): solution

        Returns:
            float: fitness value
        '''
        dim = len(x)
        exponent = 6./(dim-1) * np.array([i for i in range(dim)])
        ret = np.sum(10**exponent * x**2)
        return ret

    def F9(self, x:np.ndarray):
        '''F9 : Schwefel Function 1.2
        Separability    : Fully Non-separable
        Function Form   : Unimodal

        Args:
            x (np.ndarray): solution

        Returns:
            float: fitness value
        '''
        dim = len(x)
        ret = np.sum([np.sum(x[0:i+1])**2 for i in range(dim)])
        return ret

    ''' ＊＊＊＊＊  LSGO CEC'2013  ＊＊＊＊＊ '''
    ''' LSGO CEC'2013 Transformation Function '''

    def Tosz(self, x:np.ndarray):
        '''Noise generator (Create smooth local irregularities)

        Args:
            x (np.ndarray): solution

        Returns:
            np.ndarray: Tosz
        '''
        # x[i]_hat  =   log|x[i]|   , x[i] != 0
        #           =   0           , otherwise
        with np.errstate(divide='ignore'):
            x_hat = np.where( x != 0. , np.log( abs(x) ), 0. )
        #   c1[i]   =   10          , x[i] > 0
        #           =   5.5         , otherwise
        c1 = np.where( x > 0. , 10., 5.5 )
        #   c2[i]   =   7.9         , x[i] > 0
        #           =   3.1         , otherwise
        c2 = np.where( x > 0. , 7.9, 3.1 )
        ret = np.sign(x) * np.exp( x_hat + 0.049*( np.sin(c1*x_hat) + np.sin(c2*x_hat) ) )
        return ret

    def Tasy(self, x:np.ndarray, beta:float):
        '''Noise generator (Break the symmetry of the symmetric functions)

        Args:
            x (np.ndarray): solution
            beta (float): Degree of noise ( 0.1 ~ 0.5 )

        Returns:
            np.ndarray: Tasy
        '''
        dim = len(x)
        # [implement 1] fast overall if
        # avoid error by errstate process when 1st access
        with np.errstate(invalid='ignore'):
            ret = np.where( x <= 0. , x , x ** ( 1. + beta * np.arange(dim) / ( dim - 1.) * x**(1/2) ))
        # [implement 2] slow each if
        # ret = np.array([ x[i]  if x[i] < 0. else ( x[i] ** ( 1. + beta * i / ( dim - 1.) * x[i]**(1/2) ) )  for i in range(dim) ])
        return ret

    def Lambda(self, alpha:int|float, dim:int=1000):
        '''Noise Generator (Create ill-conditioning)

        Args:
            alpha (int): Condition number ( 10 )
            dim (int): Lambda dimension

        Returns:
            np.ndarray: Lambda (dim × dim ndarray)
        '''
        ret = np.diag( alpha ** ( np.arange(dim) / ( 2. * ( dim - 1. ) ) ) )
        return ret


    ''' LSGO CEC'2013 Data File Import '''
    def getDatafile(self):
        folder_name = 'cec2013lsgo_datafile'
        xopt_only_fnc = ['LSGO_F1', 'LSGO_F2', 'LSGO_F3', 'LSGO_F12', 'LSGO_F15']

        if self.prob_name in xopt_only_fnc :
            # xoptのみインポート
            fnc_name = self.prob_name.split('_')[1]
            # 最適解 xopt
            self.xopt    = np.loadtxt('{}/{}-xopt.txt'.format(folder_name, fnc_name))
        else :
            # xopt, p, s, w, R25, R50, R100をインポート
            fnc_name = self.prob_name.split('_')[1]
            # 最適解 xopt
            self.xopt    = np.loadtxt('{}/{}-xopt.txt'.format(folder_name, fnc_name))
            # 次元順列 p = [1,1000] -> [0,999]
            self.p       = np.loadtxt('{}/{}-p.txt'.format(folder_name, fnc_name), delimiter=',', dtype='int') - 1
            # サブコンポーネントサイズ s
            self.s       = np.loadtxt('{}/{}-s.txt'.format(folder_name, fnc_name), dtype='int')
            # 次元重み w
            self.w       = np.loadtxt('{}/{}-w.txt'.format(folder_name, fnc_name))
            # 回転行列（25D）
            self.R25     = np.loadtxt('{}/{}-R25.txt'.format(folder_name, fnc_name), delimiter=',')
            # 回転行列（50D）
            self.R50     = np.loadtxt('{}/{}-R50.txt'.format(folder_name, fnc_name), delimiter=',')
            # 回転行列（100D）
            self.R100    = np.loadtxt('{}/{}-R100.txt'.format(folder_name, fnc_name), delimiter=',')


    ''' LSGO CEC'2013 Benchmark Function '''
    ''' Noise-On '''
    ''' ----- 1. Fully-separable Function ----- '''
    # F1 : Shifted Elliptic Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Fully-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F1_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        # Transform
        z = self.Tosz( x - self.xopt )
        # Elliptic Function
        ret = self.F8(z)
        return ret

    # F2 : Shifted Rastrigin Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Fully-Separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F2_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
            self.Lambda_alpha = self.Lambda(alpha=10, dim=self.prob_dim)
        # Transform
        z = self.Lambda_alpha @ self.Tasy( self.Tosz( x - self.xopt ), beta=0.2)
        # Rastrigin Function
        ret = self.F4(z)
        return ret

    # F3 : Shifted Ackley Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Fully-Separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F3_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
            self.Lambda_alpha = self.Lambda(alpha=10, dim=self.prob_dim)
        # Transform
        z = self.Lambda_alpha @ self.Tasy( self.Tosz( x - self.xopt ), beta=0.2)
        # Ackley Function
        ret = self.F3(z)
        return ret

    ''' ----- 2. Partially Additively Separable Function Ⅰ ----- '''
    # F4 : 7-nonseparable, 1-separable Shifted and Rotated Elliptic Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F4_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 7-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.Tosz( self.R25 @ y )
            elif self.s[i] == 50 :
                z = self.Tosz( self.R50 @ y )
            elif self.s[i] == 100 :
                z = self.Tosz( self.R100 @ y )
            c += self.s[i]
            # Elliptic Function
            ret += self.w[i] * self.F8(z)
        ### 1-separable ###
        z = self.Tosz( x[self.p[c:]] - self.xopt[self.p[c:]] )
        # Elliptic Function
        ret += self.F8(z)
        return ret

    # F5 : 7-nonseparable, 1-separable Shifted and Rotated Rastrigin Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F5_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
            self.Lambda_alpha = []
            for i in range(len(self.s)):
                self.Lambda_alpha.append( self.Lambda(alpha=10, dim=self.s[i]) )
            self.Lambda_alpha.append( self.Lambda( alpha=10, dim=len(self.p[np.sum(self.s):]) ) )
        ret, c = 0., 0
        ### 7-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R25 @ y ), beta=0.2)
            elif self.s[i] == 50 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R50 @ y ), beta=0.2)
            elif self.s[i] == 100 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R100 @ y ), beta=0.2)
            c += self.s[i]
            # Rastrigin Function
            ret += self.w[i] * self.F4(z)
        ### 1-separable ###
        z = self.Lambda_alpha[len(self.s)] @ self.Tasy( self.Tosz( x[self.p[c:]] - self.xopt[self.p[c:]] ), beta=0.2)
        # Rastrigin Function
        ret += self.F4(z)
        return ret

    # F6 : 7-nonseparable, 1-separable Shifted and Rotated Ackley Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F6_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
            self.Lambda_alpha = []
            for i in range(len(self.s)):
                self.Lambda_alpha.append( self.Lambda(alpha=10, dim=self.s[i]) )
            self.Lambda_alpha.append( self.Lambda( alpha=10, dim=len(self.p[np.sum(self.s):]) ) )
        ret, c = 0., 0
        ### 7-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R25 @ y ), beta=0.2)
            elif self.s[i] == 50 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R50 @ y ), beta=0.2)
            elif self.s[i] == 100 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R100 @ y ), beta=0.2)
            c += self.s[i]
            # Ackley Function
            ret += self.w[i] * self.F3(z)
        ### 1-separable ###
        z = self.Lambda_alpha[len(self.s)] @ self.Tasy( self.Tosz( x[self.p[c:]] - self.xopt[self.p[c:]] ), beta=0.2)
        # Ackley Function
        ret += self.F3(z)
        return ret

    # F7 : 7-nonseparable, 1-separable Shifted Schwefel Function 1.2 (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F7_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 7-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.Tasy( self.Tosz( self.R25 @ y ), beta=0.2)
            elif self.s[i] == 50 :
                z = self.Tasy( self.Tosz( self.R50 @ y ), beta=0.2)
            elif self.s[i] == 100 :
                z = self.Tasy( self.Tosz( self.R100 @ y ), beta=0.2)
            c += self.s[i]
            # Schwefel Function 1.2
            ret += self.w[i] * self.F9(z)
        ### 1-separable ###
        z = self.Tasy( self.Tosz( x[self.p[c:]] - self.xopt[self.p[c:]] ), beta=0.2)
        # Sphere Function
        ret += self.F1(z)
        return ret

    # F7M : 7-nonseparable, 1-separable Shifted Schwefel Function 1.2 (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F7M_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 7-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.Tasy( self.Tosz( self.R25 @ y ), beta=0.2)
            elif self.s[i] == 50 :
                z = self.Tasy( self.Tosz( self.R50 @ y ), beta=0.2)
            elif self.s[i] == 100 :
                z = self.Tasy( self.Tosz( self.R100 @ y ), beta=0.2)
            c += self.s[i]
            # Schwefel Function 1.2
            ret += self.w[i] * self.F9(z)
        ### 1-separable ###
        z = self.Tasy( self.Tosz( x[self.p[c:]] - self.xopt[self.p[c:]] ), beta=0.2)
        # Schwefel Function 1.2
        ret += self.F9(z)
        return ret

    ''' ----- 3. Partially Additively Separable Function Ⅱ ----- '''
    # F8 : 20-nonseparable Shifted and Rotated Elliptic Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F8_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.Tosz( self.R25 @ y )
            elif self.s[i] == 50 :
                z = self.Tosz( self.R50 @ y )
            elif self.s[i] == 100 :
                z = self.Tosz( self.R100 @ y )
            c += self.s[i]
            # Elliptic Function
            ret += self.w[i] * self.F8(z)
        return ret

    # F9 : 20-nonseparable Shifted and Rotated Rastrigin Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F9_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
            self.Lambda_alpha = []
            for i in range(len(self.s)):
                self.Lambda_alpha.append( self.Lambda(alpha=10, dim=self.s[i]) )
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R25 @ y ), beta=0.2)
            elif self.s[i] == 50 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R50 @ y ), beta=0.2)
            elif self.s[i] == 100 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R100 @ y ), beta=0.2)
            c += self.s[i]
            # Rastrigin Function
            ret += self.w[i] * self.F4(z)
        return ret

    # F10 : 20-nonseparable Shifted and Rotated Ackley Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F10_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
            self.Lambda_alpha = []
            for i in range(len(self.s)):
                self.Lambda_alpha.append( self.Lambda(alpha=10, dim=self.s[i]) )
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R25 @ y ), beta=0.2)
            elif self.s[i] == 50 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R50 @ y ), beta=0.2)
            elif self.s[i] == 100 :
                z = self.Lambda_alpha[i] @ self.Tasy( self.Tosz( self.R100 @ y ), beta=0.2)
            c += self.s[i]
            # Ackley Function
            ret += self.w[i] * self.F3(z)
        return ret

    # F11 : 20-nonseparable Shifted Schwefel Function 1.2 (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F11_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.Tasy( self.Tosz( self.R25 @ y ), beta=0.2)
            elif self.s[i] == 50 :
                z = self.Tasy( self.Tosz( self.R50 @ y ), beta=0.2)
            elif self.s[i] == 100 :
                z = self.Tasy( self.Tosz( self.R100 @ y ), beta=0.2)
            c += self.s[i]
            # Schwefel Function 1.2
            ret += self.w[i] * self.F9(z)
        return ret

    ''' ----- 4. Overlapping Function ----- '''
    # F12 : Shifted Rosenbrock Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F12_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        # Transform
        z = self.Tasy( self.Tosz( x - self.xopt ), beta=0.2)
        # Rosenbrock Function
        ret = self.F2(z)
        return ret

    # F13 : Shifted Schwefel Function 1.2 with Conforming Overlapping Subcomponents (905D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Non-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F13_NOISE(self, x):
        dim = len(x)
        # Overlap Size
        m = 5
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c-i*m:c+self.s[i]-i*m]] - self.xopt[self.p[c-i*m:c+self.s[i]-i*m]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.Tasy( self.Tosz( self.R25 @ y ), beta=0.2)
            elif self.s[i] == 50 :
                z = self.Tasy( self.Tosz( self.R50 @ y ), beta=0.2)
            elif self.s[i] == 100 :
                z = self.Tasy( self.Tosz( self.R100 @ y ), beta=0.2)
            c += self.s[i]
            # Schwefel Function 1.2
            ret += self.w[i] * self.F9(z)
        return ret

    # F14 : Shifted Schwefel Function 1.2 with Conflicting Overlapping Subcomponents (905D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Non-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F14_NOISE(self, x):
        dim = len(x)
        # Overlap Size
        m = 5
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c-i*m:c+self.s[i]-i*m]] - self.xopt[c-i*m:c+self.s[i]-i*m]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.Tasy( self.Tosz( self.R25 @ y ), beta=0.2)
            elif self.s[i] == 50 :
                z = self.Tasy( self.Tosz( self.R50 @ y ), beta=0.2)
            elif self.s[i] == 100 :
                z = self.Tasy( self.Tosz( self.R100 @ y ), beta=0.2)
            c += self.s[i]
            # Schwefel Function 1.2
            ret += self.w[i] * self.F9(z)
        return ret

    ''' ----- 5. Fully Non-separable Function ----- '''
    # F15 : Shifted Schwefel Function 1.2 (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Fully Non-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F15_NOISE(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        # Transform
        z = self.Tasy( self.Tosz( x - self.xopt ), beta=0.2)
        # Schwefel Function 1.2
        ret = self.F9(z)
        return ret


    ''' Noise-Off '''
    ''' ----- 1. Fully-separable Function ----- '''
    # F1 : Shifted Elliptic Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Fully-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F1(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        # Transform
        z =  x - self.xopt
        # Elliptic Function
        ret = self.F8(z)
        return ret

    # F2 : Shifted Rastrigin Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Fully-Separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F2(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        # Transform
        z =  x - self.xopt
        # Rastrigin Function
        ret = self.F4(z)
        return ret

    # F3 : Shifted Ackley Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Fully-Separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F3(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        # Transform
        z =  x - self.xopt
        # Ackley Function
        ret = self.F3(z)
        return ret

    ''' ----- 2. Partially Additively Separable Function Ⅰ ----- '''
    # F4 : 7-nonseparable, 1-separable Shifted and Rotated Elliptic Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F4(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 7-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.R25 @ y
            elif self.s[i] == 50 :
                z = self.R50 @ y
            elif self.s[i] == 100 :
                z = self.R100 @ y
            c += self.s[i]
            # Elliptic Function
            ret += self.w[i] * self.F8(z)
        ### 1-separable ###
        z = x[self.p[c:]] - self.xopt[self.p[c:]]
        # Elliptic Function
        ret += self.F8(z)
        return ret

    # F5 : 7-nonseparable, 1-separable Shifted and Rotated Rastrigin Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F5(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 7-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.R25 @ y
            elif self.s[i] == 50 :
                z = self.R50 @ y
            elif self.s[i] == 100 :
                z = self.R100 @ y
            c += self.s[i]
            # Rastrigin Function
            ret += self.w[i] * self.F4(z)
        ### 1-separable ###
        z = x[self.p[c:]] - self.xopt[self.p[c:]]
        # Rastrigin Function
        ret += self.F4(z)
        return ret

    # F6 : 7-nonseparable, 1-separable Shifted and Rotated Ackley Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F6(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 7-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.R25 @ y
            elif self.s[i] == 50 :
                z = self.R50 @ y
            elif self.s[i] == 100 :
                z = self.R100 @ y
            c += self.s[i]
            # Ackley Function
            ret += self.w[i] * self.F3(z)
        ### 1-separable ###
        z = x[self.p[c:]] - self.xopt[self.p[c:]]
        # Ackley Function
        ret += self.F3(z)
        return ret

    # F7 : 7-nonseparable, 1-separable Shifted Schwefel Function 1.2 (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F7(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 7-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.R25 @ y
            elif self.s[i] == 50 :
                z = self.R50 @ y
            elif self.s[i] == 100 :
                z = self.R100 @ y
            c += self.s[i]
            # Schwefel Function 1.2
            ret += self.w[i] * self.F9(z)
        ### 1-separable ###
        z = x[self.p[c:]] - self.xopt[self.p[c:]]
        # Sphere Function
        ret += self.F1(z)
        return ret


    ''' ----- 3. Partially Additively Separable Function Ⅱ ----- '''
    # F8 : 20-nonseparable Shifted and Rotated Elliptic Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F8(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.R25 @ y
            elif self.s[i] == 50 :
                z = self.R50 @ y
            elif self.s[i] == 100 :
                z = self.R100 @ y
            c += self.s[i]
            # Elliptic Function
            ret += self.w[i] * self.F8(z)
        return ret

    # F9 : 20-nonseparable Shifted and Rotated Rastrigin Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F9(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.R25 @ y
            elif self.s[i] == 50 :
                z = self.R50 @ y
            elif self.s[i] == 100 :
                z = self.R100 @ y
            c += self.s[i]
            # Rastrigin Function
            ret += self.w[i] * self.F4(z)
        return ret

    # F10 : 20-nonseparable Shifted and Rotated Ackley Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F10(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.R25 @ y
            elif self.s[i] == 50 :
                z = self.R50 @ y
            elif self.s[i] == 100 :
                z = self.R100 @ y
            c += self.s[i]
            # Ackley Function
            ret += self.w[i] * self.F3(z)
        return ret

    # F11 : 20-nonseparable Shifted Schwefel Function 1.2 (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Partially-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F11(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c:c+self.s[i]]] - self.xopt[self.p[c:c+self.s[i]]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.R25 @ y
            elif self.s[i] == 50 :
                z = self.R50 @ y
            elif self.s[i] == 100 :
                z = self.R100 @ y
            c += self.s[i]
            # Schwefel Function 1.2
            ret += self.w[i] * self.F9(z)
        return ret

    ''' ----- 4. Overlapping Function ----- '''
    # F12 : Shifted Rosenbrock Function (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Separable
    ##    Function Form    |   Multimodal
    ##  ------------------ | ----------------------------
    def LSGO_F12(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        # Transform
        z = x - self.xopt
        # Rosenbrock Function
        ret = self.F2(z)
        return ret

    # F13 : Shifted Schwefel Function 1.2 with Conforming Overlapping Subcomponents (905D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Non-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F13(self, x):
        dim = len(x)
        # Overlap Size
        m = 5
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c-i*m:c+self.s[i]-i*m]] - self.xopt[self.p[c-i*m:c+self.s[i]-i*m]]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.R25 @ y
            elif self.s[i] == 50 :
                z = self.R50 @ y
            elif self.s[i] == 100 :
                z = self.R100 @ y
            c += self.s[i]
            # Schwefel Function 1.2
            ret += self.w[i] * self.F9(z)
        return ret

    # F14 : Shifted Schwefel Function 1.2 with Conflicting Overlapping Subcomponents (905D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Non-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F14(self, x):
        dim = len(x)
        # Overlap Size
        m = 5
        if self.xopt is None :
            self.getDatafile()
        ret, c = 0., 0
        ### 20-nonseparable ###
        for i in range(len(self.s)) :
            # Sampling Vector
            y = x[self.p[c-i*m:c+self.s[i]-i*m]] - self.xopt[c-i*m:c+self.s[i]-i*m]
            # Rotation & Transform
            if self.s[i] == 25 :
                z = self.R25 @ y
            elif self.s[i] == 50 :
                z = self.R50 @ y
            elif self.s[i] == 100 :
                z = self.R100 @ y
            c += self.s[i]
            # Schwefel Function 1.2
            ret += self.w[i] * self.F9(z)
        return ret

    ''' ----- 5. Fully Non-separable Function ----- '''
    # F15 : Shifted Schwefel Function 1.2 (1000D)
    ##  ------------------ | ----------------------------
    ##     Separability    |   Fully Non-separable
    ##    Function Form    |   Unimodal
    ##  ------------------ | ----------------------------
    def LSGO_F15(self, x):
        dim = len(x)
        if self.xopt is None :
            self.getDatafile()
        # Transform
        z = x - self.xopt
        # Schwefel Function 1.2
        ret = self.F9(z)
        return ret


''' main '''
if __name__ == '__main__':
    import config as cf
    import function as fc
    import logger as lg

    cnf = cf.Configuration()
    log = lg.LogData(cnf, cnf.prob_name[0])
    fnc = fc.Function(cnf, cnf.prob_name[0])
    fnc.resetTotalEvals()