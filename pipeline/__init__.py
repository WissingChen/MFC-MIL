# build a pipeline
from .base import BasePipeline
from .mil import PoolingMILPipeline
from .transmil import TransMILPipeline
from .mil_rnn import MILRNNPipeline
from .dsmil import DSMILPipeline
from .clam import CLAMPipeline
from .dtfd import DTFDPipeline
pipeline_fns = {"base": BasePipeline, "mil": PoolingMILPipeline, 'transmil': TransMILPipeline,
                "abmil": TransMILPipeline, "dsmil": DSMILPipeline, "clam": CLAMPipeline,
"mil_rnn": MILRNNPipeline,
"dtfd": DTFDPipeline}