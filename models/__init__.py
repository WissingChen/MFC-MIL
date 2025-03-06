# build a model list
from models.max_mil import MaxMIL
from models.avg_mil import AvgMIL
from models.transmil import TransMIL
from .abmil import Attention as ABMIL
from .clam import CLAMSB, CLAMMB
from .dsmil import DSMIL
from .dtfd import DTFD

model_fns = {
    'max_mil': MaxMIL, 'avg_mil': AvgMIL,
    'transmil': TransMIL, 'abmil': ABMIL,
    'dsmil': DSMIL, "dtfd": DTFD,
    'clam_sb': CLAMSB, 'clam_mb': CLAMMB
    }
    