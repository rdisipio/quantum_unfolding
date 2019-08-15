
import numpy as np
import enum 

# DWave stuff
import dimod
import dwave_networkx as dnx
import neal
from dwave_qbsolv import QBSolv

from dwave.system import EmbeddingComposite, FixedEmbeddingComposite, TilingComposite, DWaveSampler
from dwave_tools import get_embedding_with_short_chain, get_energy, anneal_sched_custom, merge_substates

from dwave_tools import *
import decimal2binary as d2b

#########################################

class Backends(enum.IntEnum):
    undefined   = 0
    cpu         = 1
    qpu         = 2
    qpu_lonoise = 2
    qpu_hinoise = 3
    sim         = 4
    hyb         = 5
    qsolv       = 6

#~~~~~~~~~~~~~~~~~~~~~~~~~~~

class StatusCode(enum.Enum):
    unknown = 0
    success = 1
    failed  = 2

#########################################


class QUBOData(object):
    def __init__(self):
        self.x = None
        self.R = None
        self.d = None
        self.y = None

        self.x_b = None
        self.R_b = None
        self.y_b = None
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_truth(self, x):
        self.x = np.copy( x )
    
    def set_response( self, R ):
        self.R = np.copy( R )
    
    def set_data( self, d ):
        self.d = np.copy( d )
    
    def set_signal( self, y ):
        self.y = np.copy( y )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~


#########################################


class QUBOUnfolder( object ):
    def __init__(self):
        self.backend = Backends.qpu_lonoise
        self.num_reads = 5000
        

        self._encoder = d2b.BinaryEncoder()
        self._auto_scaling = 0.5

        self._data = QUBOData()

        # binary encoding
        self.rho   = 4 # number of bits
        self.rho_systs = []

        # Tikhonov regularization
        self.D      = []
        self.lmbd = 0

        # Systematics
        self.syst_range = 2. # units of standard deviation
        self.syst   = []
        self.gamma = 0

        self.n_bins_truth = 0
        self.n_bins_reco  = 0
        self.n_syst       = 0

        self._status = StatusCode.unknown
        self._hardware_sampler = None
        self._bqm     = None
        self._results = []
        self.best_fit = []

        self.solver_parameters = {
                    'num_reads': 5000,
                    'auto_scale': True,
                    'annealing_time': 20,  # default: 20 us
                    'num_spin_reversal_transforms': 2,  # default: 2
                    #'anneal_schedule': anneal_sched_custom(id=3),
                    #'chain_strength' : 50000,
                    #'chain_break_fraction':0.33,
                    #'chain_break_method':None,
                    #'postprocess':  'sampling', # seems very bad!
                    #'postprocess':  'optimization',
            }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_encoding( self, rho ):
        if isinstance(rho, int):
            n = rho
            if self._data.x.shape[0]>0:
                N = self._data.x.shape[0]
                self.rho = np.array( [n]*N )
            else:
                self.rho = rho
        else:
            self.rho = np.copy( rho )

    def get_encoding( self ):
        return self.rho
    
    def check_encoding(self):
        if isinstance(self.rho, int):
            N = self._data.x.shape[0]
            n = self.rho # e.g. 4(bits), 8(bits)
            self.rho = np.array( [n]*N )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_data(self):
        return self._data
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def add_syst_1sigma( self, h_syst : np.array, n_bits=4 ):
        '''
        :param h_syst:      systematic shifts wrt nominal
        :param n_bits:      encoding
        :param syst_range:  range of systematic variation in units of standard deviation
        '''
        self.syst.append( np.copy( h_syst ) )
        self.n_syst += 1

        self.rho_systs.append( int(n_bits) )


    def get_syst_1sigma( self, i : int):
        return self.syst[i]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_regularization( self, lmbd = 1. ):
        self.lmbd = float(lmbd)

    def get_regularization( self ):
        return self.lmbd

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_syst_penalty( self, gamma=1. ):
        self.gamma = float(gamma)
    
    def get_syst_penalty( self ):
        return self.gamma

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def convert_to_binary(self):
        '''
        auto_encode derives best-guess values for alpha_i and beta_ia
        based on a scaling parameter (e.g. +- 50% ) and the truth signal distribution
        '''

        # if encoding is still a int number, change it to array of Nbits per bin
        self.check_encoding()

        self._encoder.set_rho( self.rho )

        x_b = self._encoder.auto_encode( self._data.x, 
                                         auto_range=self._auto_scaling )

        # add systematics (if any)
        self.rho_systs = np.array( self.rho_systs, dtype='uint' )

        n_bits_syst = np.sum( self.rho_systs )
        beta_syst = np.zeros( [self.n_syst, n_bits_syst] )

        if self.n_syst > 0:
            print("DEBUG: systematics encodings:")
            print(self.rho_systs)

        for isyst in range(self.n_syst):
            n_bits = self.rho_systs[isyst]

            alpha = -self.syst_range
            self._encoder.alpha = np.append( self._encoder.alpha, [alpha] )

            for j in range(n_bits):
                a = int( np.sum(self.rho_systs[:isyst]) + j )
                w = 2*self.syst_range / float(n_bits)
                beta_syst[isyst][a] = w * np.power(2, n_bits-j-1)
            
            self._encoder.rho   = np.append( self._encoder.rho, [n_bits] )

        if self.n_syst > 0:
            print("beta_syst")
            print(beta_syst)

            n_bins   = self._encoder.beta.shape[0]
            n_bits_0 = self._encoder.beta.shape[1]

            self._encoder.beta = np.block([
                            [self._encoder.beta,                      np.zeros( [ n_bins, n_bits_syst ]) ],
                            [np.zeros( [self.n_syst, n_bits_0] ),     beta_syst ] 
                        ])

        print("INFO: alpha =", self._encoder.alpha)
        print("INFO: beta =")
        print(self._encoder.beta)
        print("INFO: x_b =", x_b)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def make_qubo_matrix(self):
    
        n_params = self.n_bins_truth + self.n_syst

        Nbins = self.n_bins_truth
        Nsyst  = self.n_syst

        # regularization (Laplacian matrix)
        self.D = d2b.laplacian( self.n_bins_truth )

        # systematics
        self.S = np.zeros( [Nbins, Nbins] )

        if self.n_syst > 0:

            # matrix of systematic shifts
            T = np.vstack( self.syst ).T
            print("INFO: matrix of systematic shifts:")
            print(T)

            # update response uber-matrix
            self._data.R = np.block([self._data.R, T])
            print("INFO: response uber-matrix:")
            print(self._data.R)

            # in case Nsyst>0, extend vectors and laplacian
            self.D = np.block([
                [self.D,                   np.zeros([Nbins, Nsyst])],
                [np.zeros([Nsyst, Nbins]), np.zeros([Nsyst, Nsyst])] 
              ])

            self.S = np.block([
                    [np.zeros([Nbins, Nbins]), np.zeros([Nbins,Nsyst])], 
                    [np.zeros([Nsyst, Nbins]), np.eye(Nsyst)]
                ])

            print("INFO: systematics penalty matrix:")
            print(self.S)
            print("INFO: systematics penalty strength:", self.gamma)

        print("INFO: Laplacian operator:")
        print(self.D)
        print("INFO: regularization strength:", self.lmbd)

        d = self._data.d
        alpha = self._encoder.alpha
        beta = self._encoder.beta
        R = self._data.R
        D = self.D
        S = self.S

        W = np.einsum( 'ij,ik', R, R ) + \
            self.lmbd*np.einsum( 'ij,ik', D, D) + \
            self.gamma*np.einsum( 'ij,ik', S, S)
        print("DEBUG: W_ij =")
        print(W)

        # Using Einstein notation

        # quadratic constraints
        Qq = 2 * np.einsum( 'jk,ja,kb->ab', W, beta, beta )
        Qq = np.triu(Qq)
        np.fill_diagonal(Qq, 0.)
        print("DEBUG: quadratic coeff Qq =")
        print(Qq)

        # linear constraints
        Ql = 2*np.einsum( 'jk,k,ja->a', W, alpha, beta ) + \
             np.einsum( 'jk,ja,ka->a', W, beta, beta ) - \
             2* np.einsum( 'ij,i,ja->a', R, d, beta )
        Ql = np.diag(Ql)
        print("DEBUG: linear coeff Ql =")
        print(Ql)

        # total coeff matrix:
        self.Q = Qq + Ql

        print("DEBUG: matrix of QUBO coefficents Q_ab =:")
        print(self.Q)
        print("INFO: size of the QUBO coeff matrix is", self.Q.shape)

        return self.Q

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def find_embedding( self, J : dict, n_tries = 5 ):
 
        embedding = get_embedding_with_short_chain(J,
                                               tries=n_tries,
                                               processor=self._hardware_sampler.edgelist,
                                               verbose=True)

        return embedding

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_config_file(self):
        config_file = "dwave.config"
        if self.backend in [Backends.qpu, Backends.qpu_lonoise ]:
            config_file = "dwave.conf.wittek-lownoise"
        elif self.backend == Backends.qpu_hinoise:
            config_file = "dwave.conf.wittek-hinoise"
        else:
            raise Exception( "ERROR: unknown QPU backend", args.backend)

        return config_file

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~
         
    def solve(self):

        self.n_bins_truth = self._data.x.shape[0]
        self.n_bins_reco  = self._data.d.shape[0]

        if not self._data.R.shape[1] == self.n_bins_truth:
            raise Exception( "Number of bins at truth level do not match between 1D spectrum (%i) and response matrix (%i)" % (self.n_bins_truth,self._data.R.shape[1]) ) 
        if not self._data.R.shape[0] == self.n_bins_reco:
            raise Exception( "Number of bins at reco level do not match between 1D spectrum (%i) and response matrix (%i)" % (self.n_bins_reco,self._data.R.shape[0]) ) 

        self.convert_to_binary()

        print("INFO: N bins:", self._data.x.shape[0])
        print("INFO: n-bits encoding:", self.rho)

        print("INFO: Signal truth-level x:")
        print(self._data.x)
        print("INFO: pseudo-data b:")
        print(self._data.d)
        print("INFO: Response matrix:")
        print(self._data.R)

        self.make_qubo_matrix()
        self._bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(self.Q)

        print("INFO: solving the QUBO model (size=%i)..." % len(self._bqm))
        
        if self.backend in [ Backends.cpu ]:
            print("INFO: running on CPU...")
            self._results = dimod.ExactSolver().sample(self._bqm)
            self._status = StatusCode.success
        
        elif self.backend in [ Backends.sim ]:
            print("INFO: running on simulated annealer (neal)")

            sampler = neal.SimulatedAnnealingSampler()
            num_reads = self.solver_parameters['num_reads']
            self._results = sampler.sample( self._bqm, num_reads=num_reads).aggregate()
            self._status = StatusCode.success

        elif self.backend in [ Backends.qpu, Backends.qpu_hinoise, Backends.qpu_lonoise, Backends.hyb, Backends.qsolv ]:
            print("INFO: running on QPU")

            config_file=self.get_config_file()
            self._hardware_sampler = DWaveSampler(config_file=config_file )
            print("INFO: QPU configuration file:", config_file)

            print("INFO: finding optimal minor embedding...")

            n_tries = 5 # this might depend on encoding i.e. number of bits
            
            J = qubo_quadratic_terms_from_np_array(Q)
            embedding = self.find_embedding( J, n_tries )

            print("INFO: creating DWave sampler...")
            sampler = FixedEmbeddingComposite(self._hardware_sampler, embedding)

            if self.backend in [ Backends.qpu, Backends.qpu_hinoise, Backends.qpu_lonoise ]:
                print("INFO: Running on QPU")
                self._results = sampler.sample( self._bqm, **self.solver_parameters).aggregate()
                self._status = StatusCode.success
            
            elif self.backend in [ Backends.hyb ]:
                print("INFO: hybrid execution")
                import hybrid

                num_reads = self.solver_parameters['num_reads']
                # Define the workflow
                # hybrid.EnergyImpactDecomposer(size=len(bqm), rolling_history=0.15)
                iteration = hybrid.RacingBranches(
                    hybrid.InterruptableTabuSampler(),
                    hybrid.EnergyImpactDecomposer(size=len(bqm)//2, rolling=True)
                    | hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=num_reads)
                    | hybrid.SplatComposer()
                ) | hybrid.ArgMin()
                workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=3)
                #workflow = hybrid.Loop(iteration, max_iter=20, convergence=3)

                init_state = hybrid.State.from_problem(bqm)
                self._results = workflow.run(init_state).result().samples
                self._status = StatusCode.success

                # show execution profile
                print("INFO: timing:")
                workflow.timers
                hybrid.print_structure(workflow)
                hybrid.profiling.print_counters(workflow)

            elif self.backend in [ Backends.qsolv ]:
                print("INFO: using QBsolve with FixedEmbeddingComposite")
                self._results = QBSolv().sample_qubo(S, solver=sampler, solver_limit=5)
                self._status = StatusCode.success

            else:
                raise Exception("ERROR: unknown backend", self.backend)

        print("DEBUG: status =", self._status)
        return self._status

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_unfolded(self):

        if self._status == StatusCode.unknown:
            raise Exception( "QUBO not executed yet.")
        
        if not self._status == StatusCode.success:
            raise Exception( "QUBO not execution failed.")

        self.best_fit = self._results.first

        q = np.array(list(self.best_fit.sample.values()))
        
        self._data.y = self._encoder.decode( q )

        return self._data.y

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

