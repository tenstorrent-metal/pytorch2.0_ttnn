import os
import argparse
import torch
import torchvision

from utils import get_model, model_example_inputs, do_model_backward

def get_ttnn_backend(model_name, trace_orig, trace_modi, backward, out_folder, graphviz, device):
    use_tracer = (trace_orig or trace_modi)
    if use_tracer:
        tracer_option = {
            "model_name": model_name,
            "out_folder": out_folder,
            "trace_orig": trace_orig,
            "trace_modi": trace_modi
        }
    else:
        tracer_option = None
    option = torch_ttnn.TenstorrentBackendOption(device=device, tracer_option = tracer_option)
    return torch.compile(m, backend="ttnn", options=option)

def get_dummy_backend(model_name, trace_orig, trace_modi, backward, out_folder):
    from torch._dynamo.backends.common import aot_autograd
    def dummy_backend(gm, example_inputs):
        return gm
    def dummy_backend_with_tracer(out_prefix):
        from tracer.tracer import Tracer
        return Tracer(dummy_backend, out_prefix = out_prefix, out_folder = out_folder, \
                      trace_orig = trace_orig, trace_modi = trace_modi)
    use_tracer = (trace_orig or trace_modi)
    if not use_tracer and not backward:
        return aot_autograd(fw_compiler = dummy_backend)
    elif not use_tracer and backward:
        return aot_autograd(fw_compiler = dummy_backend, bw_compiler = dummy_backend)
    elif use_tracer and not backward:
        return aot_autograd(fw_compiler = dummy_backend_with_tracer(f"fw_{model_name}"))
    elif use_tracer and backward:
        return aot_autograd(fw_compiler = dummy_backend_with_tracer(f"fw_{model_name}"),
                            bw_compiler = dummy_backend_with_tracer(f"bw_{model_name}"))


def run_model(model_name: str, use_torch_ttnn: bool, trace_orig: bool, trace_modi, \
              backward: bool, out_folder: str, graphviz: bool, to_profile: bool, device = None):
    m = get_model(model_name)
    if m is None:
        print(f"Skip this model: {model_name}")
        return

    if backward:
        try:
            m.train()
        except:
            print(f"{model_name} Cannot to the training mode, use just eval mode")
            m.eval()
            backward = False
    else:
        m.eval()

    if use_torch_ttnn:
        backend = get_ttnn_backend(model_name, trace_orig, trace_modi, backward, out_folder, graphviz, device)
    else:
        backend = get_dummy_backend(model_name, trace_orig, trace_modi, backward, out_folder)
    m = torch.compile(m, backend = backend)

    inputs = model_example_inputs(model_name, backward)

    if to_profile:
        from torch.profiler import profile, record_function, ProfilerActivity
        trace_file = os.path.join(out_folder, "profile", model_name)
        os.makedirs(os.path.dirname(trace_file), exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, record_shapes=True) as prof:
            result = m(*inputs)
            if backward:
                do_model_backward(model_name, result)
        prof.export_chrome_trace(trace_file)
    else:
        result = m(*inputs)
        if backward:
            do_model_backward(model_name, result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", "-o", type = str, default = os.path.join(os.getcwd(),"stat"))
    parser.add_argument("--use_torch_ttnn", action="store_true")
    parser.add_argument("--trace_orig", action="store_true")
    parser.add_argument("--trace_modi", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--graphviz", action="store_true")
    parser.add_argument("--model_more", action="store_true")
    parser.add_argument("--model_sow2_list1", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.use_torch_ttnn and args.backward:
        assert(0 and "torch_ttnn not yet support backward")
    if not args.use_torch_ttnn and args.graphviz:
        assert(0 and "graphviz is in ttnn")

    if args.use_torch_ttnn:
        import torch_ttnn

    models = ["dinov2_vits14", "alexnet", "googlenet", "resnet18", "vgg11"]
    if args.model_more:
        # There have some weired bug if squentially run the model seems because of torch.SymInt
        # torch._dynamo.exc.TorchRuntimeError:
        # Failed running call_function
        # <built-in method tensor of type object at 0x7f0ff3bff9e0>(*(0.00125*s9,), **{}):
        models = ["dinov2_vits14"] + torchvision.models.list_models()
    elif args.model_sow2_list1:
        models = ["detr_resnet50",
                  "retinanet_resnet50_fpn", "mobilenet_v2", "mobilenet_v3_small",
                  "vit_b_16", "resnet50", "regnet_x_16gf",
                  "ssd300_vgg16", "ssdlite320_mobilenet_v3_large", "swin_v2_b"]

    device = torch_ttnn.ttnn.open(0) if args.use_torch_ttnn else None
    for m in models:
        try:
            run_model(m, args.use_torch_ttnn, args.trace_orig, args.trace_modi, args.backward, args.out_folder, args.graphviz, args.profile, device)
        except:
            print(f"{m} FAIL")
    if args.use_torch_ttnn:
        torch_ttnn.ttnn.close(device)
