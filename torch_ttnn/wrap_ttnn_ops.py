import torch

# Wrap functions so that they can be used in torch.fx as leaf nodes,
# and block the trace through them. See:
# https://pytorch.org/docs/stable/fx.html#torch.fx.wrap

import ttnn

# We have to create a variable for the wrapped function,
# because the `wrap` actually will try to find the target
# with "string" to find the corresponding function.
# If we register the function by `torch.fx.wrap(ttnn.add)`
# then the string been used to find the function is `add`(ya, the behavior is weird).
# But there is no variable named `add` in this file, so during compilation
# it will cause an error:
#
#       File "lib/python3.8/site-packages/torch/fx/_symbolic_trace.py", line 837, in trace
#         _patch_wrapped_functions(patcher)
#       File "lib/python3.8/site-packages/torch/fx/_symbolic_trace.py", line 1040, in _patch_wrapped_functions
#         orig_fn = frame_dict[name]
#     torch._dynamo.exc.BackendCompilerFailed: backend='compiler_fn' raised:
#     KeyError: 'add'


add = ttnn.add
matmul = ttnn.matmul
sub = ttnn.sub
mul = ttnn.mul
softmax = ttnn.softmax
tanh = ttnn.tanh
reshape = ttnn.reshape
permute = ttnn.permute

torch.fx.wrap(ttnn.add)
torch.fx.wrap(ttnn.matmul)
torch.fx.wrap(ttnn.sub)
torch.fx.wrap(ttnn.mul)
torch.fx.wrap(ttnn.softmax)
torch.fx.wrap(ttnn.tanh)
torch.fx.wrap(ttnn.reshape)
torch.fx.wrap(ttnn.permute)
