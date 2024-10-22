import copy
import os
import shutil
import sys
from collections import namedtuple
from datetime import datetime

import comfy.model_management as mm
import comfy.utils
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from einops import repeat
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)

from musepose.models.pose_guider import PoseGuider
from musepose.models.unet_2d_condition import UNet2DConditionModel
from musepose.models.unet_3d import UNet3DConditionModel
from musepose.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from musepose.utils.util import get_fps, read_frames, save_videos_grid

device_auto = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"@@device:{device_auto}")


def check_and_download():
    model_paths = [
        "MusePose/denoising_unet.pth",
        "MusePose/motion_module.pth",
        "MusePose/pose_guider.pth",
        "MusePose/reference_unet.pth",
        "dwpose/dw-ll_ucoco_384_bs5.torchscript.pt",
        "dwpose/yolox_l.torchscript.pt",
        "image_encoder/pytorch_model.bin",
        "image_encoder/config.json",
        "sd-image-variations-diffusers/unet/diffusion_pytorch_model.bin",
        "sd-image-variations-diffusers/unet/config.json",
        "sd-vae-ft-mse/diffusion_pytorch_model.bin",
        "sd-vae-ft-mse/config.json",
    ]

    for model_path in model_paths:
        local_model_path = os.path.join(PROJECT_DIR, "pretrained_weights", model_path)
        if not os.path.exists(local_model_path):
            local_base_dir = os.path.dirname(local_model_path)
            os.makedirs(local_base_dir, exist_ok=True)

            print(f"Downloading pretrained model... {model_path}")
            snapshot_download(
                repo_id="hoveyc/musepose",
                allow_patterns=[model_path],
                local_dir=os.path.join(PROJECT_DIR, "pretrained_weights"),
                local_dir_use_symlinks=False,
            )


class MusePoseGetPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "pose_images": ("IMAGE",),
                "relative_position": ("BOOLEAN", {"default": True}),
                "include_body": ("BOOLEAN", {"default": True}),
                "include_hand": ("BOOLEAN", {"default": True}),
                "include_face": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = ("poses_with_ref", "pose_images")
    FUNCTION = "process"
    CATEGORY = "musepose_list"

    def process(
        self,
        ref_image,
        pose_images,
        relative_position,
        include_body,
        include_hand,
        include_face,
    ):
        check_and_download()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        from .musepose.dwpose.dwpose_detector import DWposeDetector
        from .musepose.dwpose.util import draw_pose, draw_pose_musepose

        assert (
            ref_image.shape[1:3] == pose_images.shape[1:3]
        ), "ref_image and pose_images must have the same resolution"

        # yolo_model = "yolox_l.onnx"
        # dw_pose_model = "dw-ll_ucoco_384.onnx"
        dw_pose_model = "dw-ll_ucoco_384_bs5.torchscript.pt"
        yolo_model = "yolox_l.torchscript.pt"

        model_base_path = os.path.join(PROJECT_DIR, "pretrained_weights", "dwpose")

        model_det = os.path.join(model_base_path, yolo_model)
        model_pose = os.path.join(model_base_path, dw_pose_model)

        if not hasattr(self, "det") or not hasattr(self, "pose"):
            self.det = torch.jit.load(model_det)
            self.pose = torch.jit.load(model_pose)

            self.dwprocessor = DWposeDetector(model_det=self.det, model_pose=self.pose)

        ref_image = ref_image.squeeze(0).cpu().numpy() * 255

        self.det = self.det.to(device)
        self.pose = self.pose.to(device)

        # select ref-keypoint from reference pose for pose rescale
        ref_pose = self.dwprocessor(ref_image)
        # ref_keypoint_id = [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]
        ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        ref_keypoint_id = [
            i
            for i in ref_keypoint_id  # if ref_pose['bodies']['score'].shape[0] > 0 and ref_pose['bodies']['score'][0][i] > 0.3]
            if len(ref_pose["bodies"]["subset"]) > 0
            and ref_pose["bodies"]["subset"][0][i] >= 0.0
        ]
        ref_body = ref_pose["bodies"]["candidate"][ref_keypoint_id]

        height, width, _ = ref_image.shape
        pose_images_np = pose_images.cpu().numpy() * 255

        # read input video
        pbar = comfy.utils.ProgressBar(len(pose_images_np))
        detected_poses_np_list = []
        for img_np in pose_images_np:
            detected_poses_np_list.append(self.dwprocessor(img_np))
            pbar.update(1)

        self.det = self.det.to(offload_device)
        self.pose = self.pose.to(offload_device)

        detected_bodies = np.stack(
            [
                p["bodies"]["candidate"]
                for p in detected_poses_np_list
                if p["bodies"]["candidate"].shape[0] == 18
            ]
        )[:, ref_keypoint_id]
        # compute linear-rescale params
        if relative_position:
            ay, by = np.polyfit(
                detected_bodies[:, :, 1].flatten(),
                np.tile(ref_body[:, 1], len(detected_bodies)),
                1,
            )
            fh, fw, _ = pose_images_np[0].shape
            ax = ay / (fh / fw / height * width)
            bx = np.mean(
                np.tile(ref_body[:, 0], len(detected_bodies))
                - detected_bodies[:, :, 0].flatten() * ax
            )
        else:
            original_center_x = np.mean(detected_bodies[:, :, 0])
            original_center_y = np.mean(detected_bodies[:, :, 1])
            ax = np.mean(detected_bodies[:, :, 0].flatten()) / np.mean(ref_body[:, 0])
            ay = np.mean(detected_bodies[:, :, 1].flatten()) / np.mean(ref_body[:, 1])
            bx = (1 - ax) * original_center_x
            by = (1 - ay) * original_center_y

        a = np.array([ax, ay])
        b = np.array([bx, by])
        output_pose = []
        # pose rescale
        for detected_pose in detected_poses_np_list:
            if include_body:
                detected_pose["bodies"]["candidate"] = (
                    detected_pose["bodies"]["candidate"] * a + b
                )
            if include_face:
                detected_pose["faces"] = detected_pose["faces"] * a + b
            if include_hand:
                detected_pose["hands"] = detected_pose["hands"] * a + b
            im = draw_pose_musepose(
                detected_pose,
                height,
                width,
                include_body=include_body,
                include_hand=include_hand,
                include_face=include_face,
            )

            output_pose.append(np.array(im))

        output_pose_tensors = [torch.tensor(np.array(im)) for im in output_pose]
        output_tensor = torch.stack(output_pose_tensors) / 255

        ref_pose_img = draw_pose_musepose(
            ref_pose,
            height,
            width,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
        )

        ref_pose_tensor = torch.tensor(np.array(ref_pose_img)) / 255
        output_tensor = torch.cat((ref_pose_tensor.unsqueeze(0), output_tensor))
        output_tensor = output_tensor.permute(0, 2, 3, 1).cpu().float()

        return output_tensor, output_tensor[1:]


def scale_video(video, width, height):
    video_reshaped = video.view(
        -1, *video.shape[2:]
    )  # [batch*frames, channels, height, width]
    scaled_video = F.interpolate(
        video_reshaped, size=(height, width), mode="bilinear", align_corners=False
    )
    scaled_video = scaled_video.view(
        *video.shape[:2], scaled_video.shape[1], height, width
    )  # [batch, frames, channels, height, width]
    scaled_video = torch.squeeze(scaled_video)
    scaled_video = scaled_video.permute(1, 2, 3, 0)

    return scaled_video


def musepose(args, image, video):
    config = OmegaConf.load(args.config)
    pretrained_base_model_path = os.path.join(
        PROJECT_DIR, config.pretrained_base_model_path
    )

    check_and_download()

    pretrained_vae_path = os.path.join(PROJECT_DIR, config.pretrained_vae_path)
    image_encoder_path = os.path.join(PROJECT_DIR, config.image_encoder_path)
    denoising_unet_path = os.path.join(PROJECT_DIR, config.denoising_unet_path)
    reference_unet_path = os.path.join(PROJECT_DIR, config.reference_unet_path)
    pose_guider_path = os.path.join(PROJECT_DIR, config.pose_guider_path)
    motion_module_path = os.path.join(PROJECT_DIR, config.motion_module_path)
    inference_config_path = os.path.join(PROJECT_DIR, config.inference_config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_path,
    ).to(device_auto, dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device_auto)

    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_base_model_path,
        motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device_auto)

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device=device_auto
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(
        dtype=weight_dtype, device=device_auto
    )

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to(device_auto, dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    def handle_single(ref_image, pose_video):
        ref_image_pil = Image.fromarray(ref_image)

        pose_list = []
        pose_tensor_list = []
        pose_images = pose_video
        src_fps = args.fps
        print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
        L = min(args.L, len(pose_images))
        pose_transform = transforms.Compose(
            [transforms.Resize((height, width)), transforms.ToTensor()]
        )
        original_width, original_height = 0, 0

        pose_images = pose_images[:: args.skip + 1]
        print("processing length:", len(pose_images))
        src_fps = src_fps // (args.skip + 1)
        print("fps", src_fps)
        L = L // ((args.skip + 1))

        for pose_image_pil in pose_images[:L]:
            pose_image_pil = Image.fromarray(pose_image_pil)
            pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_list.append(pose_image_pil)
            original_width, original_height = pose_image_pil.size
            pose_image_pil = pose_image_pil.resize((width, height))

        # repeart the last segment
        last_segment_frame_num = (L - args.S) % (args.S - args.O)
        repeart_frame_num = (args.S - args.O - last_segment_frame_num) % (
            args.S - args.O
        )
        for i in range(repeart_frame_num):
            pose_list.append(pose_list[-1])
            pose_tensor_list.append(pose_tensor_list[-1])

        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=L
        )

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)

        video = pipe(
            ref_image_pil,
            pose_list,
            width,
            height,
            len(pose_list),
            args.steps,
            args.cfg,
            generator=generator,
            context_frames=args.S,
            context_stride=1,
            context_overlap=args.O,
        ).videos
        print(video.shape)

        m1 = config.pose_guider_path.split(".")[0].split("/")[-1]
        m2 = config.motion_module_path.split(".")[0].split("/")[-1]

        res = scale_video(video[:, :, :L], original_width, original_height)
        print(res.shape)
        return (res,)

    return handle_single(image, video)


class MusePoseInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "Width": ("INT", {"default": 512}),
                "Height": ("INT", {"default": 512}),
                "frame_length": ("INT", {"default": 300}),
                "slice_frame_number": ("INT", {"default": 48}),
                "slice_overlap_frame_number": ("INT", {"default": 4}),
                "cfg": ("FLOAT", {"default": 3.5}),
                "sampling_steps": ("INT", {"default": 20}),
                "fps": ("INT", {"default": 12}),
            }
        }

    CATEGORY = "musepose_list"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "musepose_func"

    def musepose_func(
        self,
        image,
        video,
        Width,
        Height,
        frame_length,
        slice_frame_number,
        slice_overlap_frame_number,
        cfg,
        sampling_steps,
        fps,
    ):
        Param = namedtuple(
            "Param",
            ["config", "W", "H", "L", "S", "O", "cfg", "seed", "steps", "fps", "skip"],
        )
        args = Param(
            os.path.join(PROJECT_DIR, "configs/test_stage_2.yaml"),
            Width,
            Height,
            frame_length,
            slice_frame_number,
            slice_overlap_frame_number,
            cfg,
            99,
            sampling_steps,
            fps,
            1,
        )

        ref_image = 255.0 * image[0].cpu().numpy()
        ref_image = np.clip(ref_image, 0, 255).astype(np.uint8)

        video = 255.0 * video.cpu().numpy()
        video = np.clip(video, 0, 255).astype(np.uint8)

        return musepose(args, ref_image, video)


NODE_CLASS_MAPPINGS = {
    "musepose_getposes": MusePoseGetPoses,
    "musepose_inference": MusePoseInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "musepose_getposes": "MusePose GetPoses",
    "musepose_inference": "MusePose Inference",
}
