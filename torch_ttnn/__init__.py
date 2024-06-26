# The inferface of this backend

from .backend import ttnn_backend
from .backend import TenstorrentBackendOption
import torch as _torch

_torch._dynamo.backends.registry.register_backend(name="ttnn", compiler_fn=ttnn_backend)

# To wrap the ttnn ops
from . import wrap_ttnn_ops
