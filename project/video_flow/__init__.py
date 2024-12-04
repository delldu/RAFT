"""Image/Video Autops Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2024(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Sat 06 Jan 2024 05:57:17 PM CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import torch
import torch.nn.functional as F
from . import raftnet
from . import flow_viz

import todos
import pdb


def get_video_flow_model():
    """Create model."""
    model = raftnet.RAFT()

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running model on {device} ...")

    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;
    # # torch::jit::setTensorExprFuserEnabled(false);
    # todos.data.mkdir("output")
    # if not os.path.exists("output/video_flow.torch"):
    #     model.save("output/video_flow.torch")

    return model, device


def flow_predict(input_files, output_dir, horizon=None):
    from tqdm import tqdm

    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_video_flow_model()

    # load files
    image_filenames = todos.data.load_files(input_files)
    n_images = total = len(image_filenames)

    # start predict
    progress_bar = tqdm(total=n_images - 1)
    for i in range(n_images - 1):
        progress_bar.update(1)

        input_tensor1 = todos.data.load_tensor(image_filenames[i])
        input_tensor2 = todos.data.load_tensor(image_filenames[i + 1])

        B, C, H, W = input_tensor2.size()

        assert input_tensor1.size() == input_tensor2.size(), "input tensors must be have same size"

        with torch.no_grad():
            predict_tensor = model(input_tensor1.to(device), input_tensor2.to(device))

        # todos.debug.output_var("predict_tensor", predict_tensor)
        # for BOF_sintel.pth
        # tensor [flow_pre] size: [2, 2, 436, 1024], min: -34.304611, max: 27.719576, mean: -0.059822
        # for BOF_things.pth
        # tensor [predict_tensor] size: [2, 2, 432, 1024], min: -34.027592, max: 31.108303, mean: -0.055991
        # for BOF_things_288960noise.pth
        # tensor [predict_tensor] size: [2, 2, 432, 1024], min: -34.538429, max: 33.962624, mean: -0.06187
        # RAFT
        # tensor [predict_tensor] size: [2, 2, 440, 1024], min: -18.201012, max: 36.731136, mean: 0.006834

        # max_flow = predict_tensor.abs().max()
        # predict_tensor /= max_flow
        # todos.debug.output_var("predict_tensor", predict_tensor[0:1, :, :, :])

        # a008 = F.interpolate(predict_tensor[0:1, :, :, :], (512, 512), mode="bilinear", align_corners=False)
        # import numpy as np
        # np.save("/tmp/a008.npy", a008.cpu().numpy())
        # todos.debug.output_var("a008", a008)


        flow_image1, flow_image2 = flow_viz.vis_pre(predict_tensor.cpu())
        flow_image1 = F.interpolate(flow_image1, (H, W), mode="bilinear", align_corners=False)
        flow_image2 = F.interpolate(flow_image2, (H, W), mode="bilinear", align_corners=False)

        output_filename = f"{output_dir}/{os.path.basename(image_filenames[i])}"
        todos.data.save_tensor([input_tensor1, input_tensor2, flow_image1, flow_image2], output_filename, nrow=2)

    todos.model.reset_device()
