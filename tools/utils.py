import torch
import torchvision

def get_model(model_name):
    if model_name == "dinov2_vits14":
        m = torch.hub.load('facebookresearch/dinov2', model_name)
    elif model_name == "detr_resnet50":
        m = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    else:
        try:
            m = torchvision.models.get_model(model_name, pretrained=True)
        except Exception as e:
            print(e)
            try:
                m = torchvision.models.get_model(model_name, pretrained=False)
            except Exception as e:
                print(e)
                return None
    return m

def model_example_inputs(model_name, backward):
    torch.manual_seed(0)
    if backward == False:
        if model_name in ["raft_small", "raft_large"]:
            input_shapes = [1, 3, 224, 224]
            return [torch.rand(input_shapes), torch.rand(input_shapes)]
        elif model_name in ["mvit_v1_b", "mvit_v2_s"]:
            input_shapes = [1, 3, 16, 224, 224]
        elif model_name in ["mc3_18", "r2plus1d_18", "r3d_18",
                            "s3d", "swin3d_b", "swin3d_s", "swin3d_t"]:
            input_shapes = [1, 3, 224, 224, 224]
        else:
            input_shapes = [1, 3, 224, 224]
        return [torch.rand(input_shapes)]
    else:
        if model_name in ["deeplabv3_mobilenet_v3_large",
                          "deeplabv3_resnet101",
                          "deeplabv3_resnet50"]:
            # batch_size should > 1
            input_shapes = [2, 3, 224, 224]
        elif model_name in ["fasterrcnn_mobilenet_v3_large_320_fpn",
                            "fasterrcnn_mobilenet_v3_large_fpn",
                            "fasterrcnn_resnet50_fpn",
                            "fasterrcnn_resnet50_fpn_v2",
                            "fcos_resnet50_fpn",
                            "keypointrcnn_resnet50_fpn",
                            "maskrcnn_resnet50_fpn",
                            "maskrcnn_resnet50_fpn_v2",
                            "retinanet_resnet50_fpn",
                            "retinanet_resnet50_fpn_v2",
                            "ssd300_vgg16",
                            "ssdlite320_mobilenet_v3_large"]:
            batch_size = 2
            images, boxes = torch.rand(batch_size, 3, 224, 224), torch.rand(batch_size, 11, 4)
            boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
            labels = torch.randint(1, 91, (batch_size, 11))
            images = list(image for image in images)
            if model_name in ["keypointrcnn_resnet50_fpn"]:
                labels = torch.randint(1, 2, (batch_size, 11))
            targets = []
            for i in range(len(images)):
                d = {}
                d['boxes'] = boxes[i]
                d['labels'] = labels[i]
                if model_name in ["keypointrcnn_resnet50_fpn"]:
                    d['keypoints'] = torch.zeros([11, 5, 3])
                if model_name in ["maskrcnn_resnet50_fpn",
                                  "maskrcnn_resnet50_fpn_v2"]:
                    d['masks'] = torch.zeros([11, 1, 1])
                targets.append(d)
            return [images, targets]
        elif model_name in ["inception_v3"]:
            input_shapes = [1, 3, 448, 448]
        elif model_name in ["raft_small", "raft_large"]:
            input_shapes = [1, 3, 224, 224]
            return [torch.rand(input_shapes), torch.rand(input_shapes)]
        elif model_name in ["mvit_v1_b", "mvit_v2_s"]:
            input_shapes = [1, 3, 16, 224, 224]
        elif model_name in ["mc3_18", "r2plus1d_18", "r3d_18",
                            "s3d", "swin3d_b", "swin3d_s", "swin3d_t"]:
            input_shapes = [1, 3, 224, 224, 224]
        else:
            input_shapes = [1, 3, 224, 224]
        return [torch.rand(input_shapes)]


def do_model_backward(model_name, result):
    if model_name in ["inception_v3"]:
        result.logits.backward(torch.ones_like(result.logits), retain_graph=True)
        result.aux_logits.backward(torch.ones_like(result.aux_logits), retain_graph=True)
    elif isinstance(result, dict):
        for k in result.keys():
            if model_name in ["detr_resnet50",
                              "deeplabv3_mobilenet_v3_large",
                              "deeplabv3_resnet101",
                              "deeplabv3_resnet50",
                              "fasterrcnn_mobilenet_v3_large_320_fpn",
                              "fasterrcnn_mobilenet_v3_large_fpn",
                              "fasterrcnn_resnet50_fpn",
                              "fasterrcnn_resnet50_fpn_v2",
                              "fcn_resnet101",
                              "fcn_resnet50",
                              "fcos_resnet50_fpn",
                              "keypointrcnn_resnet50_fpn",
                              "maskrcnn_resnet50_fpn",
                              "maskrcnn_resnet50_fpn_v2",
                              "retinanet_resnet50_fpn",
                              "retinanet_resnet50_fpn_v2",
                              "ssd300_vgg16",
                              "ssdlite320_mobilenet_v3_large"]:
                result[k].backward(torch.ones_like(result[k]), retain_graph=True)
            else:
                result[k].backward(torch.ones_like(result[k]))
    elif isinstance(result, list):
        for r in result:
            if model_name in ["raft_large", "raft_small"]:
                r.backward(torch.ones_like(r), retain_graph=True)
            else:
                r.backward(torch.ones_like(r))
    else:
        result.backward(torch.ones_like(result))
