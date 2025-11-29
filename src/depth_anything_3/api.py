# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Depth Anything 3 API module.

This module provides the main API for Depth Anything 3, including model loading,
inference, and export capabilities. It supports both single and nested model architectures.
"""

from __future__ import annotations

import os
import time
from typing import Optional, Sequence, Any, Dict
from copy import deepcopy
import numpy as np

# Check torch import with helpful error message
try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(
        "PyTorch is not installed. Please install it first:\n\n"
        "  For CUDA (Linux/Windows with NVIDIA GPU):\n"
        "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n\n"
        "  For macOS (Apple Silicon):\n"
        "    pip install torch torchvision\n\n"
        "  For CPU only:\n"
        "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n\n"
        "See https://pytorch.org/get-started/locally/ for more installation options.\n"
    ) from e

from huggingface_hub import PyTorchModelHubMixin
from PIL import Image

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.specs import Prediction

# --- IMPORTS CRITIQUES ---
from depth_anything_3.utils.export import export
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.pose_align import align_poses_umeyama
# -------------------------

from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.dynamic_batching import get_sorted_indices_by_aspect_ratio, chunk_indices
from depth_anything_3.utils.async_exporter import AsyncExporter
from depth_anything_3.utils.prefetch_pipeline import create_pipeline

# Platform-specific optimizations
if torch.cuda.is_available():
    # Enable autotuned kernels and tensor-core friendly math
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    logger.info("CUDA detected: CUDNN benchmark + TF32 enabled")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # macOS Metal Performance Shaders optimizations
    torch.set_float32_matmul_precision("medium")
    logger.info("MPS (Metal) backend detected for macOS")
else:
    torch.backends.cudnn.benchmark = False

SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


# --- Classes Wrapper pour compatibilité PrefetchPipeline ---

class DA3InputWrapper:
    """Wraps dictionary inputs to behave like a Tensor for .to(device)."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def to(self, device, non_blocking=False):
        # Transférer uniquement les tenseurs sur le device
        new_data = {}
        for k, v in self.data.items():
            if isinstance(v, torch.Tensor):
                new_data[k] = v.to(device, non_blocking=non_blocking)
            else:
                new_data[k] = v  # Garder les métadonnées (indices, etc.)
        return DA3InputWrapper(new_data)

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)


class DA3OutputWrapper:
    """Wraps dictionary outputs to handle .cpu() call from pipeline."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def cpu(self):
        # Renvoyer un dict standard avec les tenseurs sur CPU
        return {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in self.data.items()
        }

    # --- CORRECTION: Méthodes pour se comporter comme un dict ---
    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)
    # -----------------------------------------------------------


class DA3ModelWrapper(nn.Module):
    """Adapts DepthAnything3 for the generic PrefetchPipeline."""

    def __init__(self, model: DepthAnything3, export_feat_layers, infer_gs):
        super().__init__()
        self.model = model
        self.kwargs = {'export_feat_layers': export_feat_layers, 'infer_gs': infer_gs}

    def forward(self, wrapper: DA3InputWrapper):
        # Unpack wrapper
        data = wrapper.data

        # --- FIX: Préparation des entrées (ajout dimension Batch) ---
        img = data['image']
        if img.ndim == 4:
            img = img.unsqueeze(0)  # (N,3,H,W) -> (1,N,3,H,W)

        ext = data.get('extrinsics')
        if ext is not None:
            if ext.ndim == 3:
                ext = ext.unsqueeze(0)
            ext = ext.float()

        intri = data.get('intrinsics')
        if intri is not None:
            if intri.ndim == 3:
                intri = intri.unsqueeze(0)
            intri = intri.float()

        # Run original forward
        raw_output = self.model(
            img,
            ext,
            intri,
            **self.kwargs
        )

        # Package everything needed for post-processing
        combined = {
            "raw_output": raw_output,
            "imgs_cpu": data['imgs_cpu'],
            "original_indices": data['original_indices'],
            "proc_ext": data.get('extrinsics'),
            "proc_int": data.get('intrinsics')
        }

        return DA3OutputWrapper(combined)


# -----------------------------------------------------------

class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    """
    Depth Anything 3 main API class.
    """

    _commit_hash: str | None = None  # Set by mixin when loading from Hub

    def __init__(
            self,
            model_name: str = "da3-large",
            enable_compile: bool | None = None,
            compile_mode: str = "reduce-overhead",
            batch_size: int | None = None,
            mixed_precision: bool | str | None = None,
            **kwargs
    ):
        """
        Initialize DepthAnything3 with specified preset.
        """
        super().__init__()
        self.model_name = model_name
        self.compile_mode = compile_mode
        self.batch_size = batch_size

        # Validate mixed_precision parameter
        valid_mixed_precision = [None, True, False, "auto", "fp16", "float16", "fp32", "float32", "bf16", "bfloat16"]
        if mixed_precision not in valid_mixed_precision:
            raise ValueError(f"Invalid mixed_precision value: {mixed_precision}")
        self.mixed_precision = mixed_precision

        # Auto-detect optimal compile setting based on device
        if enable_compile is None:
            if torch.cuda.is_available():
                self.enable_compile = True
                logger.info("Auto-enabling torch.compile() for CUDA")
            else:
                self.enable_compile = False
                logger.info("Auto-disabling torch.compile() for MPS/CPU (better performance without)")
        else:
            self.enable_compile = enable_compile

        # Build the underlying network
        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.model.eval()

        # Apply torch.compile() optimization with persistent caching
        if self.enable_compile and hasattr(torch, 'compile'):
            try:
                self._configure_compiler()
                logger.info(f"Compiling model with torch.compile() (mode={compile_mode})...")
                self.model = torch.compile(self.model, mode=compile_mode)
                logger.info("Model compilation successful (Lazy: will occur on first inference)")
            except Exception as e:
                logger.warning(f"Model compilation failed, falling back to eager mode: {e}")

        # Initialize processors
        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()

        # Device management (set by user)
        self.device = None

    def _configure_compiler(self):
        """Configure PyTorch Compiler for persistent caching and dynamic shapes."""
        if not hasattr(torch, "_dynamo") or not hasattr(torch, "_inductor"):
            return

        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.suppress_errors = True

        try:
            import torch._inductor.config as inductor_config
            inductor_config.fx_graph_cache = True

            default_cache_path = os.path.join(os.getcwd(), ".cache", "da3_compiler")
            cache_dir = os.environ.get("DA3_COMPILER_CACHE_DIR", default_cache_path)
            os.makedirs(cache_dir, exist_ok=True)

            if hasattr(inductor_config, "cache_dir"):
                inductor_config.cache_dir = cache_dir

            logger.info(f"Persistent compilation cache enabled at: {cache_dir}")

        except ImportError:
            logger.warning("Could not import torch._inductor.config. Persistent caching disabled.")
        except Exception as e:
            logger.warning(f"Failed to configure compiler cache: {e}")

    @torch.inference_mode()
    def forward(
            self,
            image: torch.Tensor,
            extrinsics: torch.Tensor | None = None,
            intrinsics: torch.Tensor | None = None,
            export_feat_layers: list[int] | None = None,
            infer_gs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model."""
        use_autocast, autocast_dtype = self._get_autocast_settings(image.device)

        with torch.no_grad():
            if use_autocast:
                with torch.autocast(device_type=image.device.type, dtype=autocast_dtype):
                    return self.model(image, extrinsics, intrinsics, export_feat_layers, infer_gs)
            else:
                return self.model(image, extrinsics, intrinsics, export_feat_layers, infer_gs)

    def inference(
            self,
            image: list[np.ndarray | Image.Image | str],
            extrinsics: np.ndarray | None = None,
            intrinsics: np.ndarray | None = None,
            align_to_input_ext_scale: bool = True,
            infer_gs: bool = False,
            render_exts: np.ndarray | None = None,
            render_ixts: np.ndarray | None = None,
            render_hw: tuple[int, int] | None = None,
            process_res: int = 504,
            process_res_method: str = "upper_bound_resize",
            export_dir: str | None = None,
            export_format: str = "mini_npz",
            export_feat_layers: Sequence[int] | None = None,
            conf_thresh_percentile: float = 40.0,
            num_max_points: int = 1_000_000,
            show_cameras: bool = True,
            feat_vis_fps: int = 15,
            export_kwargs: Optional[dict] = {},
            use_prefetch: bool | None = None,  # NEW PARAMETER
    ) -> Prediction:
        """
        Run inference using Input Prefetching, Dynamic Batching, Async Export, and Caching.
        """
        if "gs" in export_format:
            assert infer_gs, "must set `infer_gs=True` to perform gs-related export."
        if "colmap" in export_format:
            assert isinstance(image[0], str), "`image` must be image paths for COLMAP export."

        export_feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        # --- Auto-Detect Prefetch Strategy ---
        device = self._get_model_device()
        if use_prefetch is None:
            # Disable prefetch on MPS by default (Unified Memory contention slows it down)
            if device.type == 'mps':
                use_prefetch = False
            else:
                use_prefetch = True

        # --- Export Strategy Setup ---
        streaming_formats = []
        global_formats = []
        SAFE_STREAMING_FORMATS = ["mini_npz", "npz", "depth_vis"]

        print(f"[INFO] Export directory: {export_dir}")
        if export_dir:
            for fmt in export_format.split("-"):
                print(f"[INFO] Checking format {fmt}")
                if fmt in SAFE_STREAMING_FORMATS:
                    print(f"[INFO] Enabling async export for {fmt}")
                    streaming_formats.append(fmt)
                else:
                    print(f"[INFO] Disabling async export for {fmt}")
                    global_formats.append(fmt)

        streaming_fmt_str = "-".join(streaming_formats)
        global_fmt_str = "-".join(global_formats)

        # --- Dynamic Batching & Prefetch Setup ---
        bs = self.batch_size or len(image)
        num_images = len(image)
        sorted_indices, _ = get_sorted_indices_by_aspect_ratio(image)
        all_results = []

        logger.info(
            f"Running inference on {num_images} images (Batch: {bs}) | Async Export: {'ON' if streaming_fmt_str else 'OFF'} | Prefetch: {'ON' if use_prefetch else 'OFF (MPS Default)'}")

        exporter = AsyncExporter() if streaming_fmt_str else None

        # --- STANDARD LOOP (No Prefetch - for MPS or Explicit OFF) ---
        if not use_prefetch:
            try:
                for batch_idx_list in chunk_indices(sorted_indices, bs):
                    # a. Extract & Preprocess
                    batch_images = [image[i] for i in batch_idx_list]
                    batch_ext = extrinsics[batch_idx_list] if extrinsics is not None else None
                    batch_int = intrinsics[batch_idx_list] if intrinsics is not None else None

                    imgs_cpu, proc_ext, proc_int = self._preprocess_inputs(
                        batch_images, batch_ext, batch_int, process_res, process_res_method
                    )

                    # b. Forward
                    imgs_tensor, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, proc_ext, proc_int)
                    ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
                    raw_output = self._run_model_forward(imgs_tensor, ex_t_norm, in_t, export_feat_layers, infer_gs)

                    # c. Convert
                    batch_prediction = self._convert_to_prediction(raw_output)
                    batch_prediction = self._align_to_input_extrinsics_intrinsics(proc_ext, proc_int, batch_prediction,
                                                                                  align_to_input_ext_scale)
                    batch_prediction = self._add_processed_images(batch_prediction, imgs_cpu)

                    # d. Export
                    if exporter:
                        # Reconstruct batch_images list for filenames (safe assumption: same length/order)
                        batch_images_export = [image[i] for i in batch_idx_list]
                        batch_kwargs = deepcopy(export_kwargs)
                        exporter.submit(
                            self._handle_exports, batch_prediction, streaming_fmt_str, export_dir, batch_kwargs,
                            infer_gs, render_exts, render_ixts, render_hw, conf_thresh_percentile, num_max_points,
                            show_cameras, feat_vis_fps, process_res_method, batch_images_export
                        )

                    # e. Store
                    for local_i, global_i in enumerate(batch_idx_list):
                        single_pred = self._extract_single_prediction(batch_prediction, local_i)
                        all_results.append((global_i, single_pred))
            finally:
                if exporter: exporter.shutdown(wait=True)

        # --- PREFETCH LOOP (For CUDA/CPU) ---
        else:
            model_wrapper = DA3ModelWrapper(self, export_feat_layers, infer_gs)
            pipeline = create_pipeline(model_wrapper, device, prefetch_factor=2)

            def batch_generator():
                for batch_idx_list in chunk_indices(sorted_indices, bs):
                    batch_images = [image[i] for i in batch_idx_list]
                    batch_ext = extrinsics[batch_idx_list] if extrinsics is not None else None
                    batch_int = intrinsics[batch_idx_list] if intrinsics is not None else None

                    imgs_cpu, proc_ext, proc_int = self._preprocess_inputs(
                        batch_images, batch_ext, batch_int, process_res, process_res_method
                    )

                    if device.type in ('cuda', 'mps') and imgs_cpu.ndim == 4:
                        imgs_cpu = imgs_cpu.to(memory_format=torch.channels_last)
                    if device.type == 'cuda':
                        imgs_cpu = imgs_cpu.pin_memory()

                    yield DA3InputWrapper({
                        "image": imgs_cpu, "extrinsics": proc_ext, "intrinsics": proc_int,
                        "imgs_cpu": imgs_cpu, "original_indices": batch_idx_list, "batch_images": batch_images
                    })

            try:
                pipeline_results = pipeline.run_inference(batch_generator())
                for res_dict in pipeline_results:
                    raw_output = res_dict["raw_output"]
                    imgs_cpu = res_dict["imgs_cpu"]
                    batch_idx_list = res_dict["original_indices"]
                    proc_ext = res_dict["proc_ext"]
                    proc_int = res_dict["proc_int"]
                    batch_images = res_dict.get("batch_images", [])  # Safe get

                    batch_prediction = self._convert_to_prediction(raw_output)
                    batch_prediction = self._align_to_input_extrinsics_intrinsics(proc_ext, proc_int, batch_prediction,
                                                                                  align_to_input_ext_scale)
                    batch_prediction = self._add_processed_images(batch_prediction, imgs_cpu)

                    if exporter:
                        batch_kwargs = deepcopy(export_kwargs)
                        exporter.submit(
                            self._handle_exports, batch_prediction, streaming_fmt_str, export_dir, batch_kwargs,
                            infer_gs, render_exts, render_ixts, render_hw, conf_thresh_percentile, num_max_points,
                            show_cameras, feat_vis_fps, process_res_method, batch_images
                        )

                    for local_i, global_i in enumerate(batch_idx_list):
                        single_pred = self._extract_single_prediction(batch_prediction, local_i)
                        all_results.append((global_i, single_pred))
            finally:
                if exporter: exporter.shutdown(wait=True)

        # --- Reassembly & Global Export ---
        all_results.sort(key=lambda x: x[0])
        ordered_preds = [res[1] for res in all_results]
        final_prediction = self._collate_predictions(ordered_preds)

        if export_dir and global_fmt_str:
            logger.info(f"Exporting global formats: {global_fmt_str}")
            self._handle_exports(
                final_prediction, global_fmt_str, export_dir, export_kwargs,
                infer_gs, render_exts, render_ixts, render_hw,
                conf_thresh_percentile, num_max_points, show_cameras,
                feat_vis_fps, process_res_method, image
            )

        return final_prediction

    def _extract_single_prediction(self, pred: Prediction, index: int) -> Prediction:
        """Extract i-th element from batched Prediction."""
        init_kwargs = {}
        for field in pred.__dataclass_fields__:
            val = getattr(pred, field)
            if val is None:
                init_kwargs[field] = None
                continue

            if not isinstance(val, dict) and hasattr(val, "__getitem__") and hasattr(val, "shape") and val.shape[
                0] > index:
                init_kwargs[field] = val[index: index + 1]
            elif isinstance(val, dict):
                new_dict = {}
                for k, v in val.items():
                    if hasattr(v, "__getitem__"):
                        new_dict[k] = v[index: index + 1]
                init_kwargs[field] = new_dict
            else:
                init_kwargs[field] = val

        return Prediction(**init_kwargs)

    def _collate_predictions(self, preds: list[Prediction]) -> Prediction:
        """Merge list of Predictions into one."""
        if not preds:
            return None

        first = preds[0]
        init_kwargs = {}

        for field in first.__dataclass_fields__:
            val_0 = getattr(first, field)
            if val_0 is None:
                init_kwargs[field] = None
                continue

            if isinstance(val_0, np.ndarray):
                all_vals = [getattr(p, field) for p in preds]
                # Use robust padding concatenation
                init_kwargs[field] = self._pad_concat_numpy(all_vals)
            elif isinstance(val_0, dict):
                merged_dict = {}
                for k in val_0.keys():
                    all_sub_vals = [getattr(p, field)[k] for p in preds]
                    merged_dict[k] = self._pad_concat_numpy(all_sub_vals)
                init_kwargs[field] = merged_dict
            else:
                init_kwargs[field] = val_0

        return Prediction(**init_kwargs)

    def _pad_concat_numpy(self, arrays: list[np.ndarray], axis: int = 0) -> np.ndarray:
        """Pad and concatenate numpy arrays with different spatial dimensions."""
        if not arrays:
            return np.array([])

        first_shape = arrays[0].shape
        need_padding = False
        for arr in arrays[1:]:
            if len(arr.shape) != len(first_shape) or any(
                    arr.shape[i] != first_shape[i] for i in range(1, len(first_shape))):
                need_padding = True
                break

        if not need_padding:
            return np.concatenate(arrays, axis=axis)

        max_dims = list(first_shape)
        for arr in arrays:
            for i in range(1, len(arr.shape)):
                if i < len(max_dims):
                    max_dims[i] = max(max_dims[i], arr.shape[i])
                else:
                    max_dims.append(arr.shape[i])

        padded_arrays = []
        for arr in arrays:
            pad_width = [(0, 0)] * len(arr.shape)
            for i in range(1, len(arr.shape)):
                pad_needed = max_dims[i] - arr.shape[i]
                if pad_needed > 0:
                    pad_width[i] = (0, pad_needed)

            padded_arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
            padded_arrays.append(padded_arr)

        return np.concatenate(padded_arrays, axis=axis)

    def _handle_exports(
            self, prediction, export_format, export_dir, export_kwargs,
            infer_gs, render_exts, render_ixts, render_hw,
            conf_thresh_percentile, num_max_points, show_cameras,
            feat_vis_fps, process_res_method, image
    ):
        if "gs" in export_format:
            if infer_gs and "gs_video" not in export_format:
                export_format = f"{export_format}-gs_video"
            if "gs_video" in export_format:
                if "gs_video" not in export_kwargs:
                    export_kwargs["gs_video"] = {}
                export_kwargs["gs_video"].update(
                    {
                        "extrinsics": render_exts,
                        "intrinsics": render_ixts,
                        "out_image_hw": render_hw,
                    }
                )
        if "glb" in export_format:
            if "glb" not in export_kwargs:
                export_kwargs["glb"] = {}
            export_kwargs["glb"].update(
                {
                    "conf_thresh_percentile": conf_thresh_percentile,
                    "num_max_points": num_max_points,
                    "show_cameras": show_cameras,
                }
            )
        if "feat_vis" in export_format:
            if "feat_vis" not in export_kwargs:
                export_kwargs["feat_vis"] = {}
            export_kwargs["feat_vis"].update(
                {
                    "fps": feat_vis_fps,
                }
            )
        if "colmap" in export_format:
            if "colmap" not in export_kwargs:
                export_kwargs["colmap"] = {}
            export_kwargs["colmap"].update(
                {
                    "image_paths": image,
                    "conf_thresh_percentile": conf_thresh_percentile,
                    "process_res_method": process_res_method,
                }
            )
        self._export_results(prediction, export_format, export_dir, **export_kwargs)

    def _preprocess_inputs(
            self,
            image: list[np.ndarray | Image.Image | str],
            extrinsics: np.ndarray | None = None,
            intrinsics: np.ndarray | None = None,
            process_res: int = 504,
            process_res_method: str = "upper_bound_resize",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Preprocess input images using input processor."""
        start_time = time.time()
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res,
            process_res_method,
        )
        end_time = time.time()
        logger.info(
            "Processed Images Done taking",
            end_time - start_time,
            "seconds. Shape: ",
            imgs_cpu.shape,
        )
        return imgs_cpu, extrinsics, intrinsics

    def _prepare_model_inputs(
            self,
            imgs_cpu: torch.Tensor,
            extrinsics: torch.Tensor | None,
            intrinsics: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Prepare tensors for model input."""
        device = self._get_model_device()

        if device.type == 'cuda' and imgs_cpu.device.type == 'cpu':
            imgs_cpu = imgs_cpu.pin_memory()
            if extrinsics is not None and extrinsics.device.type == 'cpu':
                extrinsics = extrinsics.pin_memory()
            if intrinsics is not None and intrinsics.device.type == 'cpu':
                intrinsics = intrinsics.pin_memory()

        if device.type in ('cuda', 'mps') and imgs_cpu.ndim == 4:
            imgs_cpu = imgs_cpu.to(memory_format=torch.channels_last)

        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()

        ex_t = (
            extrinsics.to(device, non_blocking=True)[None].float()
            if extrinsics is not None
            else None
        )
        in_t = (
            intrinsics.to(device, non_blocking=True)[None].float()
            if intrinsics is not None
            else None
        )

        return imgs, ex_t, in_t

    def _run_batched_forward(
            self,
            imgs_cpu: torch.Tensor,
            extrinsics: torch.Tensor | None,
            intrinsics: torch.Tensor | None,
            export_feat_layers: Sequence[int] | None,
            infer_gs: bool,
    ) -> dict[str, torch.Tensor]:
        """Legacy method for backward compatibility."""
        num_imgs = imgs_cpu.shape[0]
        bs = self.batch_size or num_imgs
        if bs >= num_imgs:
            imgs, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
            ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
            return self._run_model_forward(imgs, ex_t_norm, in_t, export_feat_layers, infer_gs)

        outputs: list[dict[str, torch.Tensor]] = []
        for start in range(0, num_imgs, bs):
            end = min(start + bs, num_imgs)
            imgs_slice = imgs_cpu[start:end]
            ex_slice = extrinsics[start:end] if extrinsics is not None else None
            in_slice = intrinsics[start:end] if intrinsics is not None else None

            imgs, ex_t, in_t = self._prepare_model_inputs(imgs_slice, ex_slice, in_slice)
            ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
            outputs.append(self._run_model_forward(imgs, ex_t_norm, in_t, export_feat_layers, infer_gs))

        return self._concat_model_outputs(outputs)

    def _concat_model_outputs(self, outputs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Concatenate list of model outputs along the image dimension (dim=1)."""
        if len(outputs) == 1:
            return outputs[0]
        merged: dict[str, torch.Tensor] = {}
        keys = set().union(*(o.keys() for o in outputs))
        for k in keys:
            vals = [o[k] for o in outputs if k in o]
            if torch.is_tensor(vals[0]) and vals[0].ndim >= 2 and vals[0].shape[0] == 1:
                merged[k] = torch.cat(vals, dim=1)
            else:
                merged[k] = vals[-1]
        return merged

    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:
        """Normalize extrinsics"""
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.median(dists)
        median_dist = torch.clamp(median_dist, min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

    def _align_to_input_extrinsics_intrinsics(
            self,
            extrinsics: torch.Tensor | None,
            intrinsics: torch.Tensor | None,
            prediction: Prediction,
            align_to_input_ext_scale: bool = True,
            ransac_view_thresh: int = 10,
    ) -> Prediction:
        """Align depth map to input extrinsics"""
        if extrinsics is None:
            return prediction
        prediction.intrinsics = intrinsics.numpy()
        _, _, scale, aligned_extrinsics = align_poses_umeyama(
            prediction.extrinsics,
            extrinsics.numpy(),
            ransac=len(extrinsics) >= ransac_view_thresh,
            return_aligned=True,
            random_state=42,
        )
        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].numpy()
            prediction.depth /= scale
        else:
            prediction.extrinsics = aligned_extrinsics
        return prediction

    def _run_model_forward(
            self,
            imgs: torch.Tensor,
            ex_t: torch.Tensor | None,
            in_t: torch.Tensor | None,
            export_feat_layers: Sequence[int] | None = None,
            infer_gs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run model forward pass."""
        device = imgs.device
        need_sync = device.type == "cuda"
        if need_sync:
            torch.cuda.synchronize(device)
        start_time = time.time()
        feat_layers = list(export_feat_layers) if export_feat_layers is not None else None
        output = self.forward(imgs, ex_t, in_t, feat_layers, infer_gs)
        if need_sync:
            torch.cuda.synchronize(device)
        end_time = time.time()
        logger.info(f"Model Forward Pass Done. Time: {end_time - start_time} seconds")
        return output

    def _convert_to_prediction(self, raw_output: dict[str, torch.Tensor]) -> Prediction:
        """Convert raw model output to Prediction object."""
        start_time = time.time()
        output = self.output_processor(raw_output)
        end_time = time.time()
        logger.info(f"Conversion to Prediction Done. Time: {end_time - start_time} seconds")
        return output

    def _add_processed_images(self, prediction: Prediction, imgs_cpu: torch.Tensor) -> Prediction:
        """Add processed images to prediction for visualization."""
        processed_imgs = imgs_cpu.permute(0, 2, 3, 1).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed_imgs = processed_imgs * std + mean
        processed_imgs = np.clip(processed_imgs, 0, 1)
        processed_imgs = (processed_imgs * 255).astype(np.uint8)
        prediction.processed_images = processed_imgs
        return prediction

    def _export_results(
            self, prediction: Prediction, export_format: str, export_dir: str, **kwargs
    ) -> None:
        """Export results to specified format and directory."""
        start_time = time.time()
        export(prediction, export_format, export_dir, **kwargs)
        end_time = time.time()
        logger.info(f"Export Results Done. Time: {end_time - start_time} seconds")

    def _get_autocast_settings(self, device: torch.device) -> tuple[bool, torch.dtype | None]:
        """Determine autocast settings."""
        if self.mixed_precision is False:
            return False, None

        if isinstance(self.mixed_precision, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }
            if self.mixed_precision in dtype_map:
                if device.type == "mps" and self.mixed_precision in ["bf16", "bfloat16"]:
                    logger.warning("bfloat16 not supported on MPS, falling back to float16.")
                    return True, torch.float16
                return True, dtype_map[self.mixed_precision]
            else:
                logger.warning(f"Unknown mixed precision dtype: {self.mixed_precision}, using auto-detect")

        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return True, dtype
        elif device.type == "mps":
            if self.mixed_precision is True:
                return True, torch.float16
            return False, None
        else:
            if self.mixed_precision is True:
                return True, torch.float16
            return False, None

    def to(self, *args, **kwargs):
        """Override to() to optimize model when moving to GPU devices."""
        result = super().to(*args, **kwargs)
        device = args[0] if args else kwargs.get('device')
        if device is not None:
            device_type = device if isinstance(device, str) else device.type
            if device_type in ('cuda', 'mps'):
                try:
                    self.model = self.model.to(memory_format=torch.channels_last)
                    logger.info(f"Model converted to channels_last format for {device_type}")
                except Exception as e:
                    logger.warning(f"Failed to convert to channels_last: {e}")
        return result

    def _get_model_device(self) -> torch.device:
        """Get the device where the model is located."""
        if self.device is not None:
            return self.device
        for param in self.parameters():
            self.device = param.device
            return param.device
        for buffer in self.buffers():
            self.device = buffer.device
            return buffer.device
        raise ValueError("No tensor found in model")