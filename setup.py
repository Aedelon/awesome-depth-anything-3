"""
Setup script for Depth Anything 3.
Handles post-installation of gsplat which requires torch to be installed first.
"""
import subprocess
import sys
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        install_gsplat()


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        install_gsplat()


def install_gsplat():
    """Install gsplat after torch is available."""
    print("\n" + "=" * 60)
    print("Installing gsplat (requires torch)...")
    print("=" * 60)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"
        ])
        print("✓ gsplat installed successfully")
    except subprocess.CalledProcessError:
        print("⚠ Warning: gsplat installation failed (optional dependency)")
        print("  You can install it manually later with:")
        print("  pip install 'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70'")
    print("=" * 60 + "\n")


setup(
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)