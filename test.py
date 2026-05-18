import importlib, torch
m = importlib.import_module('torch.utils._pytree')
import torchvision, transformers, sys
print('python', sys.version)
print('torch', torch.__version__, getattr(torch, '__file__', ''))
print('torchvision', torchvision.__version__)
print('transformers', transformers.__version__)
print('pytree register:', hasattr(m, 'register_pytree_node'), ' _register:', hasattr(m,'_register_pytree_node'))
