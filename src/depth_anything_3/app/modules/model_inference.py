# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# Optimizations (c) Delanoe Pirard / Aedelon - Apache 2.0
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
Model inference module for Depth Anything 3 Gradio app.

This module handles all model-related operations including inference,
data processing, and result preparation.

Optimizations based on benchmarks:
- Smart batch sizing per model/device (MPS: B=4 for small/base, B=2 for large, B=1 for giant)
- CUDA: Adaptive batching at 85% memory utilization
- CPU: Always batch=1
- Model caching for 200x faster subsequent loads
"""

import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.memory import cleanup_cuda_memory
from depth_anything_3.utils.export.glb import export_to_glb
from depth_anything_3.utils.export.gs import export_to_gs_video

# Available models for UI selection
AVAILABLE_MODELS = {
    "da3-small": "Small (fastest, ~27 img/s)",
    "da3-base": "Base (fast, ~10 img/s)",
    "da3-large": "Large (balanced, ~4 img/s)",
    "da3-giant": "Giant (high quality, ~1.6 img/s)",
    "da3nested-giant-large": "Giant+Large (best quality, ~1.5 img/s)",
}

# Mapping from UI names to HuggingFace repo IDs
MODEL_TO_HF_REPO = {
    "da3-small": "depth-anything/DA3-SMALL",
    "da3-base": "depth-anything/DA3-BASE",
    "da3-large": "depth-anything/DA3-LARGE",
    "da3-giant": "depth-anything/DA3-GIANT",
    "da3nested-giant-large": "depth-anything/DA3NESTED-GIANT-LARGE",
}

DEFAULT_MODEL = "da3nested-giant-large"


class ModelInference:
    """
    Handles model inference and data processing for Depth Anything 3.

    Uses benchmark-optimized batch sizes:
    - MPS: B=4 for small/base, B=2 for large, B=1 for giant
    - CUDA: Adaptive batching (85% VRAM utilization)
    - CPU: B=1 always
    """

    def __init__(self):
        """Initialize the model inference handler."""
        self.model: Optional[DepthAnything3] = None
        self.current_model_name: Optional[str] = None
        self.device: Optional[torch.device] = None

    def _get_optimal_batch_size(
        self, num_images: int, model_name: str, device_type: str
    ) -> int:
        """
        Get optimal batch size based on benchmarks.

        Benchmark results (MPS, 1280x720):
        - da3-small: B=4 → 27.2 img/s (vs B=1 → 22.2 img/s)
        - da3-base:  B=4 → 11.6 img/s (vs B=1 → 10.7 img/s)
        - da3-large: B=2 → 3.8 img/s  (B=4 slower due to memory pressure)
        - da3-giant: B=1 → 1.6 img/s  (B=4 → 1.2 img/s, worse!)

        Args:
            num_images: Number of images to process
            model_name: Name of the model
            device_type: Device type ('cuda', 'mps', 'cpu')

        Returns:
            Optimal batch size
        """
        if device_type == "cpu":
            return 1

        # MPS: Use benchmark-optimized fixed batch sizes
        if device_type == "mps":
            if "small" in model_name:
                return min(4, num_images)
            elif "base" in model_name:
                return min(4, num_images)
            elif "giant" in model_name:
                return 1
            else:  # large
                return min(2, num_images)

        # CUDA: Conservative batch size, can be tuned
        if "giant" in model_name:
            return min(2, num_images)
        elif "large" in model_name:
            return min(4, num_images)
        else:
            return min(8, num_images)

    def initialize_model(self, device: torch.device, model_name: str = None) -> None:
        """
        Initialize the DepthAnything3 model.

        Args:
            device: Device to load the model on
            model_name: Model name to load (default: da3-base)
        """
        if model_name is None:
            model_name = os.environ.get("DA3_MODEL_NAME", DEFAULT_MODEL)

        # Check if we need to reload the model
        need_reload = (
            self.model is None
            or self.current_model_name != model_name
            or self.device != device
        )

        if need_reload:
            # Cleanup old model if exists
            if self.model is not None:
                print(f"[ModelInference] Unloading {self.current_model_name}")
                del self.model
                self.model = None
                cleanup_cuda_memory()

            # Get HuggingFace repo ID from model name
            hf_repo = MODEL_TO_HF_REPO.get(model_name, model_name)
            print(f"[ModelInference] Loading model: {model_name} ({hf_repo}) on {device}")
            start_time = time.time()

            # Use from_pretrained to load from HuggingFace
            self.model = DepthAnything3.from_pretrained(hf_repo)
            self.model = self.model.to(device)
            self.current_model_name = model_name
            self.device = device

            load_time = time.time() - start_time
            print(f"[ModelInference] Model loaded in {load_time:.2f}s")
        else:
            print(f"[ModelInference] Reusing cached model: {model_name}")

        self.model.eval()

    def run_inference(
        self,
        target_dir: str,
        filter_black_bg: bool = False,
        filter_white_bg: bool = False,
        process_res_method: str = "upper_bound_resize",
        show_camera: bool = True,
        save_percentage: float = 30.0,
        num_max_points: int = 1_000_000,
        infer_gs: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        gs_trj_mode: str = "extend",
        gs_video_quality: str = "high",
        model_name: str = None,
    ) -> Tuple[Any, Dict[int, Dict[str, Any]]]:
        """
        Run DepthAnything3 model inference on images.

        All images are processed in a single batch for optimal performance.

        Args:
            target_dir: Directory containing images
            filter_black_bg: Whether to filter black background
            filter_white_bg: Whether to filter white background
            process_res_method: Method for resizing input images
            show_camera: Whether to show camera in 3D view
            save_percentage: Percentage of points to save (0-100)
            num_max_points: Maximum number of points in point cloud
            infer_gs: Whether to infer 3D Gaussian Splatting
            ref_view_strategy: Reference view selection strategy
            gs_trj_mode: Trajectory mode for 3DGS
            gs_video_quality: Video quality for 3DGS
            model_name: Model to use (default: da3-base)

        Returns:
            Tuple of (prediction, processed_data)
        """
        inference_start = time.time()
        print(f"[ModelInference] Processing images from {target_dir}")

        # Device check - support CUDA, MPS (Apple Silicon), and CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Initialize model (with caching)
        if model_name is None:
            model_name = DEFAULT_MODEL
        self.initialize_model(device, model_name)

        # Get image paths
        image_folder_path = os.path.join(target_dir, "images")
        all_image_paths = sorted(glob.glob(os.path.join(image_folder_path, "*")))

        # Filter for image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        image_paths = [
            path
            for path in all_image_paths
            if any(path.lower().endswith(ext) for ext in image_extensions)
        ]

        num_images = len(image_paths)
        print(f"[ModelInference] Found {num_images} images")

        if num_images == 0:
            raise ValueError("No images found. Check your upload.")

        # Map UI options to actual method names
        method_mapping = {"high_res": "lower_bound_resize", "low_res": "upper_bound_resize"}
        actual_method = method_mapping.get(process_res_method, "upper_bound_crop")

        # Get optimal batch size based on benchmarks
        batch_size = self._get_optimal_batch_size(num_images, model_name, device.type)
        print(
            f"[ModelInference] Batched inference: model={model_name}, "
            f"device={device.type}, images={num_images}, batch_size={batch_size}"
        )

        # Run model inference with batching
        with torch.no_grad():
            if num_images <= batch_size:
                # Single batch - process all at once
                prediction = self.model.inference(
                    image_paths,
                    export_dir=None,
                    process_res_method=actual_method,
                    infer_gs=infer_gs,
                    ref_view_strategy=ref_view_strategy,
                )
            else:
                # Multiple batches - process in chunks and merge
                predictions = []
                for i in range(0, num_images, batch_size):
                    batch_paths = image_paths[i : i + batch_size]
                    print(f"[ModelInference] Processing batch {i // batch_size + 1}/{(num_images + batch_size - 1) // batch_size} ({len(batch_paths)} images)")
                    batch_pred = self.model.inference(
                        batch_paths,
                        export_dir=None,
                        process_res_method=actual_method,
                        infer_gs=False,  # Only infer GS on final merged result
                        ref_view_strategy=ref_view_strategy,
                    )
                    predictions.append(batch_pred)

                # Merge all batch predictions
                prediction = self._merge_predictions(predictions)
        # num_max_points: int = 1_000_000,
        export_to_glb(
            prediction,
            filter_black_bg=filter_black_bg,
            filter_white_bg=filter_white_bg,
            export_dir=target_dir,
            show_cameras=show_camera,
            conf_thresh_percentile=save_percentage,
            num_max_points=int(num_max_points),
        )

        # export to gs video if needed
        if infer_gs:
            mode_mapping = {"extend": "extend", "smooth": "interpolate_smooth"}
            print(f"GS mode: {gs_trj_mode}; Backend mode: {mode_mapping[gs_trj_mode]}")
            export_to_gs_video(
                prediction,
                export_dir=target_dir,
                chunk_size=4,
                trj_mode=mode_mapping.get(gs_trj_mode, "extend"),
                enable_tqdm=True,
                vis_depth="hcat",
                video_quality=gs_video_quality,
            )

        # Save predictions.npz for caching metric depth data
        self._save_predictions_cache(target_dir, prediction)

        # Process results
        processed_data = self._process_results(target_dir, prediction, image_paths)

        # Clean up using centralized memory utilities for consistency with backend
        cleanup_cuda_memory()

        inference_time = time.time() - inference_start
        throughput = num_images / inference_time if inference_time > 0 else 0
        print(
            f"[ModelInference] Completed in {inference_time:.2f}s "
            f"({throughput:.1f} img/s)"
        )

        return prediction, processed_data

    def _merge_predictions(self, predictions: List[Any]) -> Any:
        """
        Merge multiple batch predictions into a single prediction.

        Args:
            predictions: List of Prediction objects from batch inference

        Returns:
            Merged Prediction object
        """
        if not predictions:
            return None
        if len(predictions) == 1:
            return predictions[0]

        from depth_anything_3.specs import Prediction

        # Concatenate arrays from all predictions
        merged_depth = np.concatenate([p.depth for p in predictions], axis=0)
        merged_conf = (
            np.concatenate([p.conf for p in predictions], axis=0)
            if predictions[0].conf is not None
            else None
        )
        merged_processed_images = (
            np.concatenate([p.processed_images for p in predictions], axis=0)
            if predictions[0].processed_images is not None
            else None
        )
        merged_extrinsics = (
            np.concatenate([p.extrinsics for p in predictions], axis=0)
            if predictions[0].extrinsics is not None
            else None
        )
        merged_intrinsics = (
            np.concatenate([p.intrinsics for p in predictions], axis=0)
            if predictions[0].intrinsics is not None
            else None
        )

        # Create merged prediction (use is_metric from first batch)
        merged = Prediction(
            depth=merged_depth,
            is_metric=predictions[0].is_metric,
            conf=merged_conf,
            extrinsics=merged_extrinsics,
            intrinsics=merged_intrinsics,
            processed_images=merged_processed_images,
        )

        print(f"[ModelInference] Merged {len(predictions)} batches into single prediction")
        return merged

    def _save_predictions_cache(self, target_dir: str, prediction: Any) -> None:
        """
        Save predictions data to predictions.npz for caching.

        Args:
            target_dir: Directory to save the cache
            prediction: Model prediction object
        """
        try:
            output_file = os.path.join(target_dir, "predictions.npz")

            # Build save dict with prediction data
            save_dict = {}

            # Save processed images if available
            if prediction.processed_images is not None:
                save_dict["images"] = prediction.processed_images

            # Save depth data
            if prediction.depth is not None:
                save_dict["depths"] = np.round(prediction.depth, 6)

            # Save confidence if available
            if prediction.conf is not None:
                save_dict["conf"] = np.round(prediction.conf, 2)

            # Save camera parameters
            if prediction.extrinsics is not None:
                save_dict["extrinsics"] = prediction.extrinsics
            if prediction.intrinsics is not None:
                save_dict["intrinsics"] = prediction.intrinsics

            # Save to file
            np.savez_compressed(output_file, **save_dict)
            print(f"Saved predictions cache to: {output_file}")

        except Exception as e:
            print(f"Warning: Failed to save predictions cache: {e}")

    def _process_results(
        self, target_dir: str, prediction: Any, image_paths: list
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process model results into structured data.

        Args:
            target_dir: Directory containing results
            prediction: Model prediction object
            image_paths: List of input image paths

        Returns:
            Dictionary containing processed data for each view
        """
        processed_data = {}

        # Read generated depth visualization files
        depth_vis_dir = os.path.join(target_dir, "depth_vis")

        if os.path.exists(depth_vis_dir):
            depth_files = sorted(glob.glob(os.path.join(depth_vis_dir, "*.jpg")))
            for i, depth_file in enumerate(depth_files):
                # Use processed images directly from API
                processed_image = None
                if prediction.processed_images is not None and i < len(
                    prediction.processed_images
                ):
                    processed_image = prediction.processed_images[i]

                processed_data[i] = {
                    "depth_image": depth_file,
                    "image": processed_image,
                    "original_image_path": image_paths[i] if i < len(image_paths) else None,
                    "depth": prediction.depth[i] if i < len(prediction.depth) else None,
                    "intrinsics": (
                        prediction.intrinsics[i]
                        if prediction.intrinsics is not None and i < len(prediction.intrinsics)
                        else None
                    ),
                    "mask": None,  # No mask information available
                }

        return processed_data

    # cleanup() removed: call cleanup_cuda_memory() directly where needed.
