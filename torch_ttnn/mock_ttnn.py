import torch


############################################################
# Device related functions
############################################################
class Device:
    def __init__(self, device_id):
        self.device_id: int = device_id

    #  def __repr__(self):
    #      """
    #      This function is necessary for torch.fx.graph._register_custom_builtin.
    #      The generated code will use this function to generate the code.
    #      The compiler must use `torch.fx.graph._register_custom_builtin`
    #      to register the custom builtin.
    #
    #      However, the ttnn module's Device does not provide a __repr__ function.
    #      So the mock here can not provide a __repr__ function either.
    #      We have to find other way. See `torch_ttnn.py` for details.
    #      """
    #      return f"Device({self.device_id})"


def open_device(device_id):
    print(f"Device {device_id} is opened")
    return Device(device_id)


def close_device(device):
    print(f"Device {device.device_id} is closed")
    pass


def from_torch(tensor):
    return tensor


def to_torch(tensor):
    return tensor


def from_device(tensor):
    print(f"Tensor with shape {tensor.shape} is moved from device")
    return tensor


def to_device(tensor, device):
    print(f"Tensor with shape {tensor.shape} is moved to device {device.device_id}")
    return tensor


def to_layout(tensor, layout):
    print(f"Tensor with shape {tensor.shape} is convert to layout {layout}")
    return tensor


############################################################
# Operations
############################################################


@torch.fx.wrap
def add(x, y):
    z = x + y
    return z


@torch.fx.wrap
def matmul(x, y):
    mm = torch.ops.aten.mm(x, y)
    return mm


@torch.fx.wrap
def sub(x, y):
    z = x - y
    return z


@torch.fx.wrap
def mul(x, y):
    z = x * y
    return z


@torch.fx.wrap
def softmax(x, axis):
    r = torch.softmax(x, axis)
    return r


@torch.fx.wrap
def tanh(x):
    r = torch.tanh(x)
    return r


@torch.fx.wrap
def reshape(x, new_shape):
    r = torch.reshape(x, new_shape)
    return r


@torch.fx.wrap
def permute(x, order):
    r = torch.permute(x, order)
    return r


ROW_MAJOR_LAYOUT = 0
TILE_LAYOUT = 1

# Wrap the functions so that they can be used in torch.fx
# and block the further recusive tracing. See:
# https://pytorch.org/docs/stable/fx.html#torch.fx.wrap
