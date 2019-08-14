
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

class Backends(enum.Enum):
    undefined   = 0
    qpu         = 1
    qpu_lonoise = 1
    qpu_hinoise = 2
    sim         = 3
    hyb         = 4
    qsolv       = 5

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
        self.alpha = [] # offset
        self.beta  = [] # scaling

        # Tikhonov regularization
        self.D      = []
        self.lmbd = 0

        # Systematics
        self.syst   = []
        self.gamma = 0

        self.n_bins_truth = 0
        self.n_bins_reco  = 0
        self.n_syst       = 0

        self._status = StatusCode.unknown
        self._hardware_sampler = None
        self._bqm     = None
        self._results = []

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_encoding( self, beta : np.array ):
        self.rho = np.copy( beta )

    def get_encoding( self ):
        return self.rho
    
    def check_encoding(self):
        if isinstance(self.rho, int):
            N = self.n_bins_truth
            n = self.rho # e.g. 4(bits), 8(bits)
            self.rho = np.array( [n]*N )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_data(self):
        return self._data
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_syst_1sigma( self, h_syst : np.array ):
        self.syst = np.copy( h_syst )
        self.n_syst = self.syst.shape[0]
    
    def get_syst_1sigma( self ):
        return self.syst
    
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

        self._data.x_b = self._encoder.auto_encode( self._data.x, 
                                                    auto_range=self._auto_scaling )


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def make_qubo_matrix(self):
        n_params = self.n_bins_truth + self.n_syst

        Nbins = self.n_bins_truth
        Nsyst  = self.n_syst

        self.Q = np.zeros( [n_params, n_params ])

        # regularization (Laplacian matrix)
        self.D = d2b.laplacian( self.n_bins_truth )

        # systematics
        self.S = np.zeros( [Nbins, Nbins] )
        if self.n_syst > 0:
            self.S = np.block([
                    [np.zeros([Nbins, Nbins]), np.zeros([Nbins,Nsyst])], 
                    [np.zeros([Nsyst, Nbins]), np.eye(Nsyst)]
                ])
            S = self.gamma * S

            # in case Nsyst>0, extend vectors and laplacian
            self.D = np.block([
                [self.D,                   np.zeros([Nbins, Nsyst])],
                [np.zeros([Nsyst, Nbins]), np.zeros([Nsyst, Nsyst])] 

              ])

        d = self._data.d
        alpha = self.alpha
        beta = self.beta
        R = self._data.R

        W = np.zeros( [Nbins, Nbins] )
        for j in range(Nbins):
            for k in range(j+1, Nbins):
                for i in range(Nbins):
                    W[j][k] += R[i][j]*R[i][k] + \
                            self.lmbd*self.D[i][j]*self.D[i][k]
        
        n_bits_tot = sum( self.rho )

        # quadratic constraints
        J = {}
        for a in range(n_bits_tot):
            for b in range(a+1, n_bits_tot):
                idx = (a, b)
                J[idx] = 0
                for j in range(Nbins):
                    for k in range(Nbins):
                        J[idx] += 2*W[j][k]*beta[j][a]*beta[k][b]

        
        # linear constraints
        h = {}
        for a in range(n_bits_tot):
            idx = (a)
            h[idx] = 0
        for a in range(n_bits_tot):
            idx = (a)
            for j in range(Nbins):
                for k in range(Nbins):
                    h[idx] += (
                        2 * W[j][k] * alpha[k] * beta[j][a] + \
                        W[j][k]*beta[j][a]*beta[k][a] )
                    for i in range(Nbins):
                        h[idx] -= 2 * R[i][j]*d[i]*beta[j][a]

        
        return h, J

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def find_embedding( self, J : dict, n_tries = 5 ):
 
        embedding = get_embedding_with_short_chain(J,
                                               tries=ntries,
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

        if not self.R.shape[1] == self.n_bins_truth:
            raise Exception( f"Number of bins at truth level do not match between 1D spectrum ({self.n_bins_truth}) and response matrix ({self.R.shape[1]})" ) 
        if not self.R.shape[0] == self.n_bins_reco:
            raise Exception( f"Number of bins at reco level do not match between 1D spectrum ({self.n_bins_reco}) and response matrix ({self.R.shape[0]})" ) 

        self.check_encoding()
        self.convert_to_binary()

        print("INFO: N bins:", N)
        print("INFO: n-bits encoding:", n)

        print("INFO: Signal truth-level x:")
        print(self.x)
        print("INFO: pseudo-data b:")
        print(self.d)
        print("INFO: Response matrix:")
        print(self.R)
        print("INFO: Laplacian operator:")
        print(self.D)
        print("INFO: regularization strength:", self.lmbd)

        h, J = self.make_qubo_matrix()

        self._bqm = dimod.BinaryQuadraticModel( linear=h,
                                                quadratic=J,
                                                offset=0.0,
                                                vartype=dimod.BINARY)

        print("INFO: solving the QUBO model (size=%i)..." % len(self._bqm))

        if self.backend in ['cpu']:
            print("INFO: running on CPU...")
            self._results = dimod.ExactSolver().sample(self._bqm)
            self._status = StatusCode.success
        
        elif self.backend in ['sim']:
            print("INFO: running on simulated annealer (neal)")

            sampler = neal.SimulatedAnnealingSampler()

            self._results = sampler.sample( self._bqm, num_reads=num_reads).aggregate()
            self._status = StatusCode.success

        elif self.backend in [ Backends.qpu, Backends.qpu_hinoise, Backends.qpu_lonoise, Backends.hyb, Backends.qsolv ]:
            print("INFO: running on QPU")

            self._hardware_sampler = DWaveSampler(config_file=self.get_config_file() )
            print("INFO: QPU configuration file:", config_file)

            print("INFO: finding optimal minor embedding...")

            ntries = 5 # this might depend on encoding i.e. number of bits
            
            embedding = self.find_embedding( J, n_tries )

            print("INFO: creating DWave sampler...")
            sampler = FixedEmbeddingComposite(self._hardware_sampler, embedding)

            solver_parameters = {
                    'num_reads': num_reads,
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

            if self.backend in [ Backends.qpu, Backends.qpu_hinoise, Backends.qpu_lonoise ]:
                print("INFO: Running on QPU")
                self._results = sampler.sample( self._bqm, **solver_parameters).aggregate()
                self._status = StatusCode.success
            
            elif self.backend in [ Backends.hyb ]:
                print("INFO: hybrid execution")
                import hybrid

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

        return self._status

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_unfolded(self):

        if self._status == 0:
            raise Exception( "QUBO not executed yet.")

        else:
            raise Exception( "QUBO not execution failed.")

        best_fit = self._results.first

        q = np.array(list(best_fit.sample.values()))

        self._data.y = self._encoder.decode( q )

        return self.y

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

