from torch.nn import CrossEntropyLoss as CE
# from .LDAMLoss import LDAMLoss as LDAM
from .FocalLoss import FocalLoss as Focal


# # Automatically create a list of all classes imported in this file
# import sys
# import inspect
# LOSSES = [name for name, obj in sys.modules[__name__].__dict__.items() if inspect.isclass(obj)]
# print(f'{LOSSES = }')