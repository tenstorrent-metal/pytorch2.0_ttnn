import os
import argparse
import torch
import torchvision

def run_model(model_name: str, backend: str, backward: bool, out_path: str, graphviz: bool, to_profile: bool, device = None):
    if model_name == "dinov2_vits14":
        m = torch.hub.load('facebookresearch/dinov2', model_name)
    else:
        try:
            m = torchvision.models.get_model(model_name, pretrained=True)
        except Exception as e:
            print(e)
            try:
                m = torchvision.models.get_model(model_name, pretrained=False)
            except Exception as e:
                print(e)
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
    if backend == "torch_ttnn":
        option = torch_ttnn.TorchTtnnOption(device=device)
        m = torch.compile(m, backend=torch_ttnn.backend(option))
    elif backend == "torch_stat":
        option = torch_stat.TorchStatOption(model_name = model_name, backward = backward,
                                            out = out_path, gen_graphviz=graphviz)
        m = torch.compile(m, backend=torch_stat.backend(option))
    else:
        assert(0 and "Unsupport backend")

    inputs = [torch.randn([1, 3, 224, 224])]

    if to_profile:
        from torch.profiler import profile, record_function, ProfilerActivity
        trace_file = os.path.join(out_path, "profile", model_name)
        os.makedirs(os.path.dirname(trace_file), exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, record_shapes=True) as prof:
            result = m(*inputs)
            if backward:
                result.backward(torch.ones_like(result))
        prof.export_chrome_trace(trace_file)
    else:
        result = m(*inputs)
        if backward:
            result.backward(torch.ones_like(result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", "-o", type = str, default = os.path.join(os.getcwd(),"stat"))
    parser.add_argument("--backend", type = str)
    parser.add_argument("--graphviz", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert(args.backend in ["torch_ttnn", "torch_stat"])
    if args.backend == "torch_ttnn" and args.backward:
        assert(0 and "torch_ttnn not yet support backward")

    if args.backend == "torch_ttnn":
        import torch_ttnn
    elif args.backend == "torch_stat":
        from torch_ttnn import torch_stat

    models = ["dinov2_vits14", "alexnet", "googlenet", "resnet18", "vgg11"]

    device = torch_ttnn.ttnn.open_device(device_id = 0) if args.backend == "torch_ttnn" else None
    for m in models:
        try:
            run_model(m, args.backend, args.backward, args.out_path, args.graphviz, args.profile, device)
        except:
            print(f"{m} FAIL")
    if args.backend == "torch_ttnn":
        torch_ttnn.ttnn.close_device(device)