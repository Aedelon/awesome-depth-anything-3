#!/usr/bin/env python3
# Copyright (c) Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0
"""
Hugging Face Spaces entry point for awesome-depth-anything-3.

This file is the main entry point for the HF Spaces deployment.
It launches the Gradio web interface with optimized settings for cloud deployment.
"""

import os
import tempfile

# Disable analytics and configure for HF Spaces
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["DA3_LOG_LEVEL"] = "WARNING"

from depth_anything_3.app.gradio_app import DepthAnything3App


def main():
    """Launch the Gradio app for HF Spaces."""
    # Use DA3-LARGE for good balance of quality and speed
    workspace_dir = "/tmp/workspace"
    gallery_dir = "/tmp/gallery"

    # Create directories
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)

    app = DepthAnything3App(
        model_dir="depth-anything/DA3-LARGE",
        workspace_dir=workspace_dir,
        gallery_dir=gallery_dir,
    )

    demo = app.create_app()

    # Build allowed paths for Gradio file access
    allowed_paths = [
        os.getcwd(),
        tempfile.gettempdir(),
        workspace_dir,
        gallery_dir,
        "/tmp",
    ]

    # Launch for HF Spaces (theme/css already set in create_app via gr.Blocks())
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        allowed_paths=allowed_paths,
    )


if __name__ == "__main__":
    main()
