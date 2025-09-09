# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403
from .fair import FairDataset  # noqa: F401, F403

from .dota import DOTADataset  # noqa: F401, F403
from .mask_dota import MaskDOTADataset   # noqa: F401, F403
from .occluded_dota import OccludedDOTADataset  # noqa: F401, F403

from .dior_r import DIOR_RDataset  # noqa: F401, F403
from .occluded_dior_r import OccludedDIOR_RDataset  # noqa: F401, F403

from .hrsc2016 import HRSC2016Dataset  # noqa: F401, F403
from .occluded_hrsc2016 import OccludedHRSC2016Dataset  # noqa: F401, F403

__all__ = ['build_dataset',
           'SARDataset', 'HRSCDataset', 'FairDataset',
           'DOTADataset', 'DIOR_RDataset','HRSC2016Dataset',
           'MaskDOTADataset', 'OccludedDOTADataset',
           'OccludedDIOR_RDataset', 'OccludedHRSC2016Dataset']