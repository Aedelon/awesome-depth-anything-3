"""
Prefetch pipeline for overlapping I/O and compute during inference.

Provides device-specific implementations:
- CUDA: Uses torch.cuda.Stream for async transfers + compute overlap
- MPS: CPU-side prefetch with ThreadPoolExecutor
- CPU: Memory prefetch to avoid I/O stalls

Expected gains:
- CUDA: 15-25% throughput improvement
- MPS: 10-15% throughput improvement
- CPU: 3-8% throughput improvement
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import torch
import time
from typing import Iterator, List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PrefetchMetrics:
    """Metrics collector for prefetch pipeline performance."""

    def __init__(self):
        self.total_batches = 0
        self.total_time = 0.0
        self.transfer_time = 0.0
        self.compute_time = 0.0
        self.prefetch_wait_time = 0.0
        self.errors = []

    def add_batch(self, transfer_ms: float, compute_ms: float, wait_ms: float = 0.0):
        """Record timing for a single batch."""
        self.total_batches += 1
        self.transfer_time += transfer_ms
        self.compute_time += compute_ms
        self.prefetch_wait_time += wait_ms

    def add_error(self, error: Exception):
        """Record an error."""
        self.errors.append(str(error))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if self.total_batches == 0:
            return {"error": "No batches processed"}

        return {
            "total_batches": self.total_batches,
            "total_time_s": self.total_time,
            "avg_transfer_ms": self.transfer_time / self.total_batches,
            "avg_compute_ms": self.compute_time / self.total_batches,
            "avg_wait_ms": self.prefetch_wait_time / self.total_batches,
            "throughput_batch_per_s": self.total_batches / self.total_time if self.total_time > 0 else 0,
            "overlap_efficiency": self._compute_overlap_efficiency(),
            "errors": self.errors,
        }

    def _compute_overlap_efficiency(self) -> float:
        """
        Compute overlap efficiency: how much transfer and compute overlap.
        1.0 = perfect overlap, 0.0 = no overlap (sequential)
        """
        if self.total_batches == 0 or self.total_time == 0:
            return 0.0

        # Sequential time = sum of all operations
        sequential_time = self.transfer_time + self.compute_time

        # Actual time is measured total_time
        # Overlap efficiency = (sequential - actual) / sequential
        if sequential_time == 0:
            return 0.0

        saved_time = sequential_time - (self.total_time * 1000)  # total_time is in seconds
        return max(0.0, min(1.0, saved_time / sequential_time))


class PrefetchPipeline(ABC):
    """Abstract pipeline for overlapping I/O and compute."""

    def __init__(self, model: torch.nn.Module, device: torch.device, prefetch_factor: int = 2):
        """
        Initialize prefetch pipeline.

        Args:
            model: PyTorch model for inference
            device: Target device (cuda/mps/cpu)
            prefetch_factor: Number of batches to prefetch (default: 2)
        """
        self.model = model
        self.device = device
        self.prefetch_factor = prefetch_factor
        self.metrics = PrefetchMetrics()

    @abstractmethod
    def run_inference(self, batch_loader: Iterator) -> List[torch.Tensor]:
        """
        Run inference with prefetch.

        Args:
            batch_loader: Iterator yielding batches

        Returns:
            List of output tensors
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.get_summary()


class CUDAPrefetchPipeline(PrefetchPipeline):
    """Pipeline optimized for NVIDIA GPUs with CUDA Streams."""

    def __init__(self, model: torch.nn.Module, device: torch.device, prefetch_factor: int = 2):
        super().__init__(model, device, prefetch_factor)
        self.transfer_stream = torch.cuda.Stream()
        logger.info(f"CUDA prefetch pipeline initialized (prefetch_factor={prefetch_factor})")

    def run_inference(self, batch_loader: Iterator) -> List[torch.Tensor]:
        """
        Run inference with CUDA stream-based prefetch.

        Uses dual streams:
        - Main stream: compute (forward pass)
        - Transfer stream: async CPU->GPU transfers
        """
        results = []
        batch_iter = iter(batch_loader)
        start_time = time.perf_counter()

        try:
            # Preload first batch
            batch_current = next(batch_iter).to(self.device, non_blocking=True)
            torch.cuda.current_stream().synchronize()

            for batch_next_cpu in batch_iter:
                batch_start = time.perf_counter()

                # Prefetch batch N+1 on dedicated stream
                transfer_start = time.perf_counter()
                with torch.cuda.stream(self.transfer_stream):
                    batch_next = batch_next_cpu.to(self.device, non_blocking=True)
                transfer_time = (time.perf_counter() - transfer_start) * 1000

                # Compute batch N on main stream (overlaps with transfer)
                compute_start = time.perf_counter()
                with torch.no_grad():
                    output = self.model(batch_current)
                    results.append(output.cpu())
                compute_time = (time.perf_counter() - compute_start) * 1000

                # Sync before swap
                torch.cuda.current_stream().wait_stream(self.transfer_stream)
                batch_current = batch_next

                self.metrics.add_batch(transfer_time, compute_time)

            # Last batch
            compute_start = time.perf_counter()
            with torch.no_grad():
                output = self.model(batch_current)
                results.append(output.cpu())
            compute_time = (time.perf_counter() - compute_start) * 1000
            self.metrics.add_batch(0.0, compute_time)

        except Exception as e:
            logger.error(f"Error in CUDA prefetch pipeline: {e}")
            self.metrics.add_error(e)
            raise RuntimeError(f"CUDA prefetch pipeline failed: {e}") from e

        finally:
            self.metrics.total_time = time.perf_counter() - start_time

        return results


class MPSPrefetchPipeline(PrefetchPipeline):
    """Pipeline for Apple Silicon with CPU-side prefetch."""

    def __init__(self, model: torch.nn.Module, device: torch.device, prefetch_factor: int = 2):
        super().__init__(model, device, prefetch_factor)
        self.prefetch_queue = Queue(maxsize=prefetch_factor)
        logger.info(f"MPS prefetch pipeline initialized (prefetch_factor={prefetch_factor})")

    def _prefetch_worker(self, batch_loader: Iterator):
        """
        Worker thread to preload batches.

        Runs in background to keep queue filled while main thread processes batches.
        """
        try:
            for batch in batch_loader:
                self.prefetch_queue.put(batch)
        except Exception as e:
            logger.error(f"Error in MPS prefetch worker: {e}")
            self.metrics.add_error(e)
            self.prefetch_queue.put(e)  # Signal error to main thread
        finally:
            self.prefetch_queue.put(None)  # Sentinel for end of stream

    def run_inference(self, batch_loader: Iterator) -> List[torch.Tensor]:
        """
        Run inference with CPU-side prefetch.

        Background thread loads batches while main thread transfers and computes.
        """
        results = []
        start_time = time.perf_counter()

        try:
            # Launch prefetch worker in background
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._prefetch_worker, batch_loader)

                while True:
                    # Get next batch from queue
                    wait_start = time.perf_counter()
                    try:
                        batch = self.prefetch_queue.get(timeout=30.0)
                    except Empty:
                        raise TimeoutError("Prefetch queue timeout (30s)")
                    wait_time = (time.perf_counter() - wait_start) * 1000

                    if batch is None:  # End of stream
                        break

                    if isinstance(batch, Exception):  # Worker error
                        raise batch

                    # Transfer + compute
                    transfer_start = time.perf_counter()
                    batch_mps = batch.to(self.device, non_blocking=True)
                    transfer_time = (time.perf_counter() - transfer_start) * 1000

                    compute_start = time.perf_counter()
                    with torch.no_grad():
                        output = self.model(batch_mps)
                        results.append(output.cpu())
                    compute_time = (time.perf_counter() - compute_start) * 1000

                    self.metrics.add_batch(transfer_time, compute_time, wait_time)

                # Check for worker exceptions
                future.result()

        except Exception as e:
            logger.error(f"Error in MPS prefetch pipeline: {e}")
            self.metrics.add_error(e)
            raise RuntimeError(f"MPS prefetch pipeline failed: {e}") from e

        finally:
            self.metrics.total_time = time.perf_counter() - start_time

        return results


class CPUPrefetchPipeline(PrefetchPipeline):
    """Pipeline for CPU with memory prefetch."""

    def __init__(self, model: torch.nn.Module, device: torch.device, prefetch_factor: int = 2):
        super().__init__(model, device, prefetch_factor)
        self.prefetch_queue = Queue(maxsize=prefetch_factor)
        logger.info(f"CPU prefetch pipeline initialized (prefetch_factor={prefetch_factor})")

    def _prefetch_worker(self, batch_loader: Iterator):
        """
        Worker thread to preload batches into RAM.

        Avoids I/O stalls during inference by loading ahead.
        """
        try:
            for batch in batch_loader:
                # Force load into RAM (clone to ensure detached from I/O)
                batch_loaded = batch.clone() if torch.is_tensor(batch) else batch
                self.prefetch_queue.put(batch_loaded)
        except Exception as e:
            logger.error(f"Error in CPU prefetch worker: {e}")
            self.metrics.add_error(e)
            self.prefetch_queue.put(e)
        finally:
            self.prefetch_queue.put(None)

    def run_inference(self, batch_loader: Iterator) -> List[torch.Tensor]:
        """
        Run inference with memory prefetch.

        Background thread loads batches into RAM ahead of inference.
        """
        results = []
        start_time = time.perf_counter()

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._prefetch_worker, batch_loader)

                while True:
                    wait_start = time.perf_counter()
                    try:
                        batch = self.prefetch_queue.get(timeout=30.0)
                    except Empty:
                        raise TimeoutError("Prefetch queue timeout (30s)")
                    wait_time = (time.perf_counter() - wait_start) * 1000

                    if batch is None:
                        break

                    if isinstance(batch, Exception):
                        raise batch

                    # Compute (no transfer for CPU)
                    compute_start = time.perf_counter()
                    with torch.no_grad():
                        output = self.model(batch)
                        results.append(output)
                    compute_time = (time.perf_counter() - compute_start) * 1000

                    self.metrics.add_batch(0.0, compute_time, wait_time)

                future.result()

        except Exception as e:
            logger.error(f"Error in CPU prefetch pipeline: {e}")
            self.metrics.add_error(e)
            raise RuntimeError(f"CPU prefetch pipeline failed: {e}") from e

        finally:
            self.metrics.total_time = time.perf_counter() - start_time

        return results


def create_pipeline(
    model: torch.nn.Module,
    device: torch.device,
    prefetch_factor: int = 2
) -> PrefetchPipeline:
    """
    Factory to create appropriate pipeline for device.

    Args:
        model: PyTorch model for inference
        device: Target device (cuda/mps/cpu)
        prefetch_factor: Number of batches to prefetch (default: 2)

    Returns:
        Device-specific prefetch pipeline

    Examples:
        >>> device = torch.device('mps')
        >>> pipeline = create_pipeline(model, device, prefetch_factor=2)
        >>> results = pipeline.run_inference(batch_loader)
        >>> metrics = pipeline.get_metrics()
    """
    if device.type == 'cuda':
        return CUDAPrefetchPipeline(model, device, prefetch_factor)
    elif device.type == 'mps':
        return MPSPrefetchPipeline(model, device, prefetch_factor)
    else:  # cpu
        return CPUPrefetchPipeline(model, device, prefetch_factor)
