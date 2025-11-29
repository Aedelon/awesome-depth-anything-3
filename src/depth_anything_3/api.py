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

import time
from typing import Optional, Sequence
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
from depth_anything_3.utils.export import export
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.pose_align import align_poses_umeyama
from depth_anything_3.utils.async_exporter import AsyncExporter
from depth_anything_3.utils.dynamic_batching import get_sorted_indices_by_aspect_ratio, chunk_indices

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

# torch.compile() optimizations
if hasattr(torch, '_dynamo'):
    # Capture scalar outputs to reduce graph breaks
    torch._dynamo.config.capture_scalar_outputs = True
    # Enable automatic dynamic shapes for better compilation
    torch._dynamo.config.automatic_dynamic_shapes = True
    # Suppress excessive warnings
    torch._dynamo.config.suppress_errors = False
    logger.info("torch.compile() optimizations enabled")

SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    """
    Depth Anything 3 main API class.

    This class provides a high-level interface for depth estimation using Depth Anything 3.
    It supports both single and nested model architectures with metric scaling capabilities.

    Features:
    - Hugging Face Hub integration via PyTorchModelHubMixin
    - Support for multiple model presets (vitb, vitg, nested variants)
    - Automatic mixed precision inference
    - Export capabilities for various formats (GLB, PLY, NPZ, etc.)
    - Camera pose estimation and metric depth scaling

    Usage:
        # Load from Hugging Face Hub
        model = DepthAnything3.from_pretrained("huggingface/model-name")

        # Or create with specific preset
        model = DepthAnything3(preset="vitg")

        # Run inference
        prediction = model.inference(images, export_dir="output", export_format="glb")
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

        Args:
        model_name: The name of the model preset to use.
                    Examples: 'da3-giant', 'da3-large', 'da3metric-large', 'da3nested-giant-large'.
        enable_compile: Whether to use torch.compile() for optimization (default: None = auto-detect).
                       Auto-detects: True for CUDA, False for MPS/CPU.
                       Provides 30-50% speedup on CUDA but may slow down MPS.
        compile_mode: Compilation mode for torch.compile() (default: "reduce-overhead").
                     Options:
                     - "default": Standard compilation (balanced)
                     - "reduce-overhead": Minimize Python overhead (best for inference)
                     - "max-autotune": Maximum performance tuning (slower compilation, CUDA only)
        batch_size: Batch size for processing images (default: None = process all at once).
                   Lower values reduce memory usage but may increase processing time.
        mixed_precision: Mixed precision mode (default: None = auto-detect).
                        Options:
                        - None: Auto-detect (bfloat16 on CUDA if supported, float16 otherwise)
                        - True: Enable with auto-detection
                        - False: Disable (use float32)
                        - "bfloat16": Force bfloat16
                        - "float16": Force float16
        **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
        self.model_name = model_name
        self.compile_mode = compile_mode
        self.batch_size = batch_size

        # Validate mixed_precision parameter
        valid_mixed_precision = [None, True, False, "auto", "fp16", "float16", "fp32", "float32", "bf16", "bfloat16"]
        if mixed_precision not in valid_mixed_precision:
            raise ValueError(
                f"Invalid mixed_precision value: {mixed_precision!r}\n"
                f"Valid options: {valid_mixed_precision}\n\n"
                f"Examples:\n"
                f"  - None or 'auto': Auto-detect (recommended)\n"
                f"  - True: Enable with auto-detection\n"
                f"  - False: Disable (use float32)\n"
                f"  - 'fp16' or 'float16': Force float16\n"
                f"  - 'fp32' or 'float32': Force float32\n"
                f"  - 'bf16' or 'bfloat16': Force bfloat16 (CUDA Ampere+ only)\n"
            )
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

        # Apply torch.compile() optimization if enabled and PyTorch 2.0+
        if self.enable_compile and hasattr(torch, 'compile'):
            try:
                logger.info(f"Compiling model with torch.compile() (mode={compile_mode})...")
                self.model = torch.compile(self.model, mode=compile_mode)
                logger.info("Model compilation successful")
            except Exception as e:
                logger.warning(f"Model compilation failed, falling back to eager mode: {e}")

        # Initialize processors
        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()

        # Device management (set by user)
        self.device = None

    @torch.inference_mode()
    def forward(
            self,
            image: torch.Tensor,
            extrinsics: torch.Tensor | None = None,
            intrinsics: torch.Tensor | None = None,
            export_feat_layers: list[int] | None = None,
            infer_gs: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            image: Input batch with shape ``(B, N, 3, H, W)`` on the model device.
            extrinsics: Optional camera extrinsics with shape ``(B, N, 4, 4)``.
            intrinsics: Optional camera intrinsics with shape ``(B, N, 3, 3)``.
            export_feat_layers: Layer indices to return intermediate features for.

        Returns:
            Dictionary containing model predictions
        """
        # Determine mixed precision settings
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
            # GLB export parameters
            conf_thresh_percentile: float = 40.0,
            num_max_points: int = 1_000_000,
            show_cameras: bool = True,
            # Feat_vis export parameters
            feat_vis_fps: int = 15,
            # Other export parameters
            export_kwargs: Optional[dict] = {},
    ) -> Prediction:
        """
        Run inference on input images using Dynamic Resolution Batching and Async Export.
        """
        if "gs" in export_format:
            assert infer_gs, "must set `infer_gs=True` to perform gs-related export."

        if "colmap" in export_format:
            assert isinstance(image[0], str), "`image` must be image paths for COLMAP export."

        export_feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        # --- Export Strategy Setup ---
        # Séparer les formats "streamables" (safe pour l'async par batch) des formats globaux
        streaming_formats = []
        global_formats = []

        # Liste des formats qui produisent des fichiers individuels (et donc safe pour le batch processing)
        SAFE_STREAMING_FORMATS = ["mini_npz", "npz", "depth_vis"]

        if export_dir:
            for fmt in export_format.split("-"):
                if fmt in SAFE_STREAMING_FORMATS:
                    streaming_formats.append(fmt)
                else:
                    # GLB, Video, Colmap nécessitent le contexte global ou sont des fichiers monolithiques
                    global_formats.append(fmt)

        streaming_fmt_str = "-".join(streaming_formats)
        global_fmt_str = "-".join(global_formats)

        # --- Dynamic Batching Setup ---
        bs = self.batch_size or len(image)
        num_images = len(image)
        sorted_indices, _ = get_sorted_indices_by_aspect_ratio(image)
        all_results = []

        logger.info(
            f"Running inference on {num_images} images (Batch: {bs}) | Async Export: {'ON' if streaming_fmt_str else 'OFF'}")

        # Initialiser l'exportateur asynchrone
        exporter = AsyncExporter() if streaming_fmt_str else None

        try:
            # --- Processing Loop ---
            for batch_idx_list in chunk_indices(sorted_indices, bs):
                # a. Extract batch data
                batch_images = [image[i] for i in batch_idx_list]
                batch_ext = extrinsics[batch_idx_list] if extrinsics is not None else None
                batch_int = intrinsics[batch_idx_list] if intrinsics is not None else None

                # b. Preprocess
                imgs_cpu, proc_ext, proc_int = self._preprocess_inputs(
                    batch_images, batch_ext, batch_int, process_res, process_res_method
                )

                # c. Forward
                imgs_tensor, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, proc_ext, proc_int)
                ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)

                raw_output = self._run_model_forward(
                    imgs_tensor, ex_t_norm, in_t, export_feat_layers, infer_gs
                )

                # d. Convert & Align
                batch_prediction = self._convert_to_prediction(raw_output)
                batch_prediction = self._align_to_input_extrinsics_intrinsics(
                    proc_ext, proc_int, batch_prediction, align_to_input_ext_scale
                )
                batch_prediction = self._add_processed_images(batch_prediction, imgs_cpu)

                # e. Async Export (Streaming)
                if exporter:
                    # On crée une copie des kwargs pour ce batch
                    batch_kwargs = deepcopy(export_kwargs)

                    # On soumet la tâche d'export au thread worker
                    exporter.submit(
                        self._handle_exports,
                        batch_prediction,
                        streaming_fmt_str,
                        export_dir,
                        batch_kwargs,
                        infer_gs, render_exts, render_ixts, render_hw,
                        conf_thresh_percentile, num_max_points, show_cameras,
                        feat_vis_fps, process_res_method,
                        batch_images  # Passer les images du batch pour le nommage des fichiers
                    )

                # f. Store results
                for local_i, global_i in enumerate(batch_idx_list):
                    single_pred = self._extract_single_prediction(batch_prediction, local_i)
                    all_results.append((global_i, single_pred))

        finally:
            # S'assurer que tous les exports asynchrones sont terminés
            if exporter:
                exporter.shutdown(wait=True)

        # --- Reassembly ---
        all_results.sort(key=lambda x: x[0])
        ordered_preds = [res[1] for res in all_results]
        final_prediction = self._collate_predictions(ordered_preds)

        # --- Global Export ---
        # Exporter les formats restants (GLB, Video...) qui nécessitent toutes les données
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
        """Extrait le i-ème élément d'un objet Prediction batché."""
        init_kwargs = {}
        for field in pred.__dataclass_fields__:
            val = getattr(pred, field)
            if val is None:
                init_kwargs[field] = None
                continue

            # Slicing logic
            if (not isinstance(val, dict)
                    and hasattr(val, "__getitem__")
                    and hasattr(val, "shape")
                    and val.shape[0] > index
            ):
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
        """Fusionne une liste de Predictions en une seule Prediction."""
        if not preds:
            # Retourne None ou lève une erreur appropriée si la liste est vide
            # Ici on suppose que le code appelant gère ça ou que preds n'est jamais vide
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
                init_kwargs[field] = self._pad_concat_numpy(all_vals)
            elif isinstance(val_0, dict):
                merged_dict = {}
                # Assuming all dicts have same keys
                for k in val_0.keys():
                    all_sub_vals = [getattr(p, field)[k] for p in preds]
                    merged_dict[k] = np.concatenate(all_sub_vals, axis=0)
                init_kwargs[field] = merged_dict
            else:
                # For scalars or lists that shouldn't be concatenated, take the first one
                init_kwargs[field] = val_0

        return Prediction(**init_kwargs)

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
        # Add GLB export parameters
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
        # Add Feat_vis export parameters
        if "feat_vis" in export_format:
            if "feat_vis" not in export_kwargs:
                export_kwargs["feat_vis"] = {}
            export_kwargs["feat_vis"].update(
                {
                    "fps": feat_vis_fps,
                }
            )
        # Add COLMAP export parameters
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

        # Use pinned memory for faster H2D copies on CUDA
        if device.type == 'cuda' and imgs_cpu.device.type == 'cpu':
            imgs_cpu = imgs_cpu.pin_memory()
            if extrinsics is not None and extrinsics.device.type == 'cpu':
                extrinsics = extrinsics.pin_memory()
            if intrinsics is not None and intrinsics.device.type == 'cpu':
                intrinsics = intrinsics.pin_memory()

        # Apply channels_last to CPU tensor first (if it's 4D)
        if device.type in ('cuda', 'mps') and imgs_cpu.ndim == 4:
            imgs_cpu = imgs_cpu.to(memory_format=torch.channels_last)

        # Move images to model device and add batch dimension
        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()

        # Apply channels_last to the batched tensor if needed (now it's 5D: B, N, C, H, W)
        # Note: channels_last only works for 4D tensors, so we skip it for 5D

        # Convert camera parameters to tensors
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
        """
        Run forward pass, optionally splitting the batch to save memory.
        Note: This is kept for backward compatibility but dynamic batching uses
        the loop inside inference() directly.
        """
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
                # For scalars or non-batched entries, keep the last one
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
        # Convert from (N, 3, H, W) to (N, H, W, 3) and denormalize
        processed_imgs = imgs_cpu.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)

        # Denormalize from ImageNet normalization
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
        """
        Determine autocast settings based on mixed_precision configuration.

        Args:
            device: The device where the model is running

        Returns:
            Tuple of (use_autocast, dtype)
        """
        # If mixed precision is explicitly disabled
        if self.mixed_precision is False:
            return False, None

        # If mixed precision is explicitly set to a dtype string
        if isinstance(self.mixed_precision, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }
            if self.mixed_precision in dtype_map:
                # bf16 not reliably supported on MPS; fall back to fp16 there
                if device.type == "mps" and self.mixed_precision in ["bf16", "bfloat16"]:
                    logger.warning(
                        "bfloat16 is not supported on MPS (Apple Silicon). "
                        "Falling back to float16. "
                        "To suppress this warning, use mixed_precision='float16' explicitly."
                    )
                    return True, torch.float16
                return True, dtype_map[self.mixed_precision]
            else:
                logger.warning(f"Unknown mixed precision dtype: {self.mixed_precision}, using auto-detect")

        # Auto-detect (default behavior when mixed_precision is None or True)
        if device.type == "cuda":
            # Use bfloat16 on CUDA if supported, otherwise float16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return True, dtype
        elif device.type == "mps":
            # Default to fp32 on MPS; allow opt-in fp16 when explicitly requested
            if self.mixed_precision is True:
                return True, torch.float16
            return False, None
        else:
            # CPU: optionally use float16 if explicitly enabled
            if self.mixed_precision is True:
                return True, torch.float16
            return False, None

    def to(self, *args, **kwargs):
        """
        Override to() to optimize model when moving to GPU devices.

        Applies channels_last memory format for better performance on CUDA/MPS.
        """
        result = super().to(*args, **kwargs)

        # Apply channels_last to the underlying model for GPU acceleration
        device = args[0] if args else kwargs.get('device')
        if device is not None:
            device_type = device if isinstance(device, str) else device.type
            if device_type in ('cuda', 'mps'):
                try:
                    # Convert model to channels_last for better conv performance
                    self.model = self.model.to(memory_format=torch.channels_last)
                    logger.info(f"Model converted to channels_last format for {device_type}")
                except Exception as e:
                    logger.warning(f"Failed to convert to channels_last: {e}")

        return result

    def _get_model_device(self) -> torch.device:
        """
        Get the device where the model is located.

        Returns:
            Device where the model parameters are located

        Raises:
            ValueError: If no tensors are found in the model
        """
        if self.device is not None:
            return self.device

        # Find device from parameters
        for param in self.parameters():
            self.device = param.device
            return param.device

        # Find device from buffers
        for buffer in self.buffers():
            self.device = buffer.device
            return buffer.device

        raise ValueError("No tensor found in model")

    def _pad_concat_numpy(self, arrays: list[np.ndarray], axis: int = 0) -> np.ndarray:
        """
        Remplit (pad) et concatène les tableaux numpy ayant des dimensions spatiales différentes.
        Ceci est nécessaire pour reassembler les résultats du Dynamic Batching.
        """
        if not arrays:
            # Retourne un tableau vide compatible avec l'initialisation ultérieure
            return np.array([])

        # 1. Vérification rapide si le padding est nécessaire
        need_padding = False
        first_shape = arrays[0].shape
        for arr in arrays[1:]:
            if len(arr.shape) != len(first_shape) or any(
                    arr.shape[i] != first_shape[i] for i in range(1, len(first_shape))):
                need_padding = True
                break

        if not need_padding:
            # Chemins rapide : concaténation standard si toutes les dimensions correspondent
            return np.concatenate(arrays, axis=axis)

        # 2. Trouver les dimensions maximales
        max_dims = list(first_shape)
        for arr in arrays:
            # Commence à l'index 1 (dim de batch 0 est ignorée)
            for i in range(1, len(arr.shape)):
                if i < len(max_dims):
                    max_dims[i] = max(max_dims[i], arr.shape[i])
                # Gère le cas improbable où un tableau a plus de dimensions spatiales
                else:
                    max_dims.append(arr.shape[i])

        # 3. Appliquer le padding et concaténer
        padded_arrays = []
        for arr in arrays:
            pad_width = [(0, 0)] * len(arr.shape)
            # Applique le padding de l'index 1 aux dimensions maximales
            for i in range(1, len(arr.shape)):
                pad_needed = max_dims[i] - arr.shape[i]
                if pad_needed > 0:
                    # Pad seulement à la fin (droite/bas)
                    pad_width[i] = (0, pad_needed)

                    # Utilise 'constant' (valeur par défaut 0) pour remplir l'espace
            padded_arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
            padded_arrays.append(padded_arr)

        return np.concatenate(padded_arrays, axis=axis)