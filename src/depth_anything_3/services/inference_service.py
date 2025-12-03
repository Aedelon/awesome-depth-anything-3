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
Unified Inference Service
Provides unified interface for local and remote inference
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
import typer

from ..api import DepthAnything3


class InferenceService:
    """Unified inference service class"""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.model = None

    def load_model(self):
        """Load model"""
        if self.model is None:
            typer.echo(f"Loading model from {self.model_dir}...")
            self.model = DepthAnything3.from_pretrained(self.model_dir).to(self.device)
        return self.model

    def run_local_inference(
        self,
        image_paths: List[str],
        export_dir: str,
        export_format: str = "mini_npz-glb",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers: List[int] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_input_ext_scale: bool = True,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
        batch_size: Union[int, str] = "all",
        max_batch_size: int = 64,
        target_memory_utilization: float = 0.85,
    ) -> Any:
        """Run local inference.

        Args:
            batch_size: Batch size for processing:
                - "all": Process all images in one batch (original behavior)
                - "auto": Adaptive batching based on available GPU memory
                - int: Fixed batch size (e.g., 4, 8, 16)
            max_batch_size: Maximum batch size when using adaptive batching
            target_memory_utilization: Target GPU memory usage (0.0-1.0) for adaptive batching
        """
        if export_feat_layers is None:
            export_feat_layers = []

        model = self.load_model()

        typer.echo(f"Running inference on {len(image_paths)} images...")

        # Use batch_inference for adaptive or fixed batching
        if batch_size != "all":
            typer.echo(f"Batch mode: {batch_size}" + (f" (max: {max_batch_size})" if batch_size == "auto" else ""))

            def progress_callback(processed: int, total: int):
                typer.echo(f"  Progress: {processed}/{total} images")

            predictions = model.batch_inference(
                images=image_paths,
                process_res=process_res,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
                target_memory_utilization=target_memory_utilization,
                progress_callback=progress_callback,
            )

            # For batch inference, we need to merge results and export
            # The first prediction contains all depth maps if we need to export
            if export_dir and predictions:
                # Merge all predictions into one for export
                merged = self._merge_predictions(predictions)
                from depth_anything_3.utils.export import export
                export(
                    merged,
                    export_dir,
                    export_format,
                    conf_thresh_percentile=conf_thresh_percentile,
                    num_max_points=num_max_points,
                    show_cameras=show_cameras,
                    feat_vis_fps=feat_vis_fps,
                )
                typer.echo(f"Results saved to {export_dir}")
                typer.echo(f"Export format: {export_format}")

            return predictions

        # Original behavior: process all images in one batch
        inference_kwargs = {
            "image": image_paths,
            "export_dir": export_dir,
            "export_format": export_format,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "export_feat_layers": export_feat_layers,
            "align_to_input_ext_scale": align_to_input_ext_scale,
            "use_ray_pose": use_ray_pose,
            "ref_view_strategy": ref_view_strategy,
            "conf_thresh_percentile": conf_thresh_percentile,
            "num_max_points": num_max_points,
            "show_cameras": show_cameras,
            "feat_vis_fps": feat_vis_fps,
        }

        # Add pose data (if exists)
        if extrinsics is not None:
            inference_kwargs["extrinsics"] = extrinsics
        if intrinsics is not None:
            inference_kwargs["intrinsics"] = intrinsics

        prediction = model.inference(**inference_kwargs)

        typer.echo(f"Results saved to {export_dir}")
        typer.echo(f"Export format: {export_format}")

        return prediction

    def _merge_predictions(self, predictions: List[Any]) -> Any:
        """Merge multiple batch predictions into one."""
        if not predictions:
            return None
        if len(predictions) == 1:
            return predictions[0]

        # Merge depth maps and other fields
        merged_depths = []
        merged_confs = []
        merged_image_paths = []

        for pred in predictions:
            if hasattr(pred, 'depth') and pred.depth is not None:
                # Handle batched depth maps
                if pred.depth.dim() == 4:  # (B, 1, H, W)
                    for i in range(pred.depth.shape[0]):
                        merged_depths.append(pred.depth[i])
                else:
                    merged_depths.append(pred.depth)

            if hasattr(pred, 'confidence') and pred.confidence is not None:
                if pred.confidence.dim() == 4:
                    for i in range(pred.confidence.shape[0]):
                        merged_confs.append(pred.confidence[i])
                else:
                    merged_confs.append(pred.confidence)

            if hasattr(pred, 'image_paths') and pred.image_paths:
                merged_image_paths.extend(pred.image_paths)

        # Use first prediction as base and update fields
        import torch
        merged = predictions[0]
        if merged_depths:
            merged.depth = torch.stack(merged_depths)
        if merged_confs:
            merged.confidence = torch.stack(merged_confs)
        if merged_image_paths:
            merged.image_paths = merged_image_paths

        return merged

    def run_backend_inference(
        self,
        image_paths: List[str],
        export_dir: str,
        backend_url: str,
        export_format: str = "mini_npz-glb",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers: List[int] = None,
        extrinsics: Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        align_to_input_ext_scale: bool = True,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        feat_vis_fps: int = 15,
    ) -> Dict[str, Any]:
        """Run backend inference"""
        if export_feat_layers is None:
            export_feat_layers = []

        # Check backend status
        if not self._check_backend_status(backend_url):
            raise typer.BadParameter(f"Backend service is not running at {backend_url}")

        # Prepare payload
        payload = {
            "image_paths": image_paths,
            "export_dir": export_dir,
            "export_format": export_format,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "export_feat_layers": export_feat_layers,
            "align_to_input_ext_scale": align_to_input_ext_scale,
            "use_ray_pose": use_ray_pose,
            "ref_view_strategy": ref_view_strategy,
            "conf_thresh_percentile": conf_thresh_percentile,
            "num_max_points": num_max_points,
            "show_cameras": show_cameras,
            "feat_vis_fps": feat_vis_fps,
        }

        # Add pose data (if exists)
        if extrinsics is not None:
            payload["extrinsics"] = [ext.astype(np.float64).tolist() for ext in extrinsics]
        if intrinsics is not None:
            payload["intrinsics"] = [intr.astype(np.float64).tolist() for intr in intrinsics]

        # Submit task
        typer.echo("Submitting inference task to backend...")
        try:
            response = requests.post(f"{backend_url}/inference", json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result["success"]:
                task_id = result["task_id"]
                typer.echo("Task submitted successfully!")
                typer.echo(f"Task ID: {task_id}")
                typer.echo(f"Results will be saved to: {export_dir}")
                typer.echo(f"Check backend logs for progress updates with task ID: {task_id}")
                return result
            else:
                raise typer.BadParameter(
                    f"Backend inference submission failed: {result['message']}"
                )
        except requests.exceptions.RequestException as e:
            raise typer.BadParameter(f"Backend inference submission failed: {e}")

    def _check_backend_status(self, backend_url: str) -> bool:
        """Check backend status"""
        try:
            response = requests.get(f"{backend_url}/status", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


def run_inference(
    image_paths: List[str],
    export_dir: str,
    model_dir: str,
    device: str = "cuda",
    backend_url: Optional[str] = None,
    export_format: str = "mini_npz-glb",
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    export_feat_layers: List[int] = None,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    align_to_input_ext_scale: bool = True,
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    conf_thresh_percentile: float = 40.0,
    num_max_points: int = 1_000_000,
    show_cameras: bool = True,
    feat_vis_fps: int = 15,
    batch_size: Union[int, str] = "all",
    max_batch_size: int = 64,
    target_memory_utilization: float = 0.85,
) -> Union[Any, Dict[str, Any]]:
    """Unified inference interface.

    Args:
        batch_size: Batch size for processing:
            - "all": Process all images in one batch (default, original behavior)
            - "auto": Adaptive batching based on available GPU memory
            - int: Fixed batch size (e.g., 4, 8, 16)
        max_batch_size: Maximum batch size when using adaptive batching (default: 64)
        target_memory_utilization: Target GPU memory usage (0.0-1.0) for adaptive batching
    """

    service = InferenceService(model_dir, device)

    if backend_url:
        return service.run_backend_inference(
            image_paths=image_paths,
            export_dir=export_dir,
            backend_url=backend_url,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=align_to_input_ext_scale,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
        )
    else:
        return service.run_local_inference(
            image_paths=image_paths,
            export_dir=export_dir,
            export_format=export_format,
            process_res=process_res,
            process_res_method=process_res_method,
            export_feat_layers=export_feat_layers,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=align_to_input_ext_scale,
            use_ray_pose=use_ray_pose,
            ref_view_strategy=ref_view_strategy,
            conf_thresh_percentile=conf_thresh_percentile,
            num_max_points=num_max_points,
            show_cameras=show_cameras,
            feat_vis_fps=feat_vis_fps,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            target_memory_utilization=target_memory_utilization,
        )
