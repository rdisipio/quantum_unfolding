
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~

class QUBOUnfolder( object ):
    def __init__(self):
        self.backend = Backends.qpu_lonoise
        self.num_reads = 5000
        self.encoding  = np.zeros(1)
        self.x = np.zeros(1)
        self.R0 = np.diag(1)
        self.d    = np.zeros(1)
        self.syst = np.zeros(1)
        self.y    = np.zeros(1)
        self.lmbd = 0
        self.gamma = 0

        self.n_bins_truth = 0
        self.n_bins_reco  = 0
        self.n_syst       = 0

        self._status = StatusCode.unknown
        self._hardware_sampler = None
        self._bqm     = None
        self._results = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_encoding( self, beta : np.array ):
        self.encoding = np.copy( beta )

    def get_encoding( self ):
        return self.encoding
    
    def check_encoding(self):
        if isinstance(self.encoding, int):
            n = self.encoding # e.g. 4(bits), 8(bits)
            self.encoding = np.array( [n]*self.n_bins_truth )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_truth( self, h_truth : np.array ):
        self.x = np.copy( h_truth )
        self.n_bins_truth = self.x.shape[0]
    
    def get_truth( self ):
        return self.x
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_response( self, h_response : np.array ):
        self.R0 = np.copy( h_response )
    
    def get_response( self ):
        return self.R0
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_data( self, h_data : np.array ):
        self.d = np.copy( h_data )
        self.n_bins_reco = self.d.shape[0]
    
    def get_data( self ):
        return self.d

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_syst_1sigma( self, h_syst : np.array ):
        self.syst = np.copy( h_syst )
        self.n_syst = self.syst.shape[0]
    
    def get_syst_1sigma( self ):
        return self.syst
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_regularization( self, lmbd = 1. ):
        self.lmbd = lmbd

    def get_regularization( self ):
        return self.lmbd

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_syst_penalty( self, gamma=1. ):
        self.gamma = gamma
    
    def get_syst_penalty( self ):
        return self.gamma

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def make_qubo_matrix(self):
        n_params = self.n_bins_truth + self.n_syst

        self.Q = np.zeros( [n_params, n_params ])

        # regularization (Laplacian matrix)
        D = d2b.laplacian( self.n_bins_truth )

        # linear constraints
        h = {}

        # quadratic constraints
        J = {}

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
         
    def run(self):

        if not self.R0.shape[1] == self.n_bins_truth:
            raise Exception( f"Number of bins at truth level do not match between 1D spectrum ({self.n_bins_truth}) and response matrix ({self.R0.shape[1]})" ) 
        if not self.R0.shape[0] == self.n_bins_reco:
            raise Exception( f"Number of bins at reco level do not match between 1D spectrum ({self.n_bins_reco}) and response matrix ({self.R0.shape[0]})" ) 

        self.check_encoding()

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

            self._results = sampler.sample( self._bqm, **solver_parameters).aggregate()
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

        self.y = d2b.compact_vector(q, self.encoding )

        return self.y

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~

