import dwave_networkx as dnx
import minorminer
import numpy as np
import sys
from dwave.system.composites import FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler


def max_chain_length(embedding: dict)->int:
    max_ = 0
    for _, chain in embedding.items():
        if len(chain) > max_:
            max_ = len(chain)
    return max_


def get_embedding_with_short_chain(J: dict, tries: int = 5,
                                   processor: list = None, verbose=False)->dict:
    '''Try a few probabilistic embeddings and return the one with the shortest
    chain length

    :param J: Couplings
    :param tries: Number of probabilistic embeddings
    :param verbose: Whether to print out diagnostic information

    :return: Returns the minor embedding
    '''
    if processor is None:
        # The hardware topology: 16 by 16 pieces of K_4,4 unit cells
        processor = dnx.chimera_graph(16, 16, 4).edges()
    # Try a few embeddings
    best_chain_length = sys.maxsize
    source = list(J.keys())
    for _ in range(tries):
        try:
            emb = minorminer.find_embedding(source, processor)
            chain_length = max_chain_length(emb)
            if chain_length > 0 and chain_length < best_chain_length:
                embedding = emb
                best_chain_length = chain_length
        except:
            pass
    if verbose:
        print(best_chain_length, max_chain_length(embedding))
    if best_chain_length == sys.maxsize:
        raise Exception("Cannot find embedding")
    return embedding
