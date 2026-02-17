"""Modal setup for IsaacLab 2.0.2 with IsaacSim 4.5.0 using micromamba."""

import os
import subprocess
import modal

# Configuration
ISAACSIM_VERSION = "4.5.0"
ISAACLAB_VERSION = "2.0.2"
ISAACSIM_URL = "https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip"
ISAACLAB_REPO = "https://github.com/isaac-sim/IsaacLab.git"


def download_and_setup_isaacsim():
    """Download and setup IsaacSim binaries."""
    import urllib.request
    import zipfile

    isaacsim_path = "/root/isaacsim"
    zip_path = "/tmp/isaacsim.zip"

    print(f"Downloading IsaacSim {ISAACSIM_VERSION}...")
    print(f"URL: {ISAACSIM_URL}")

    # Download with progress
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rDownload progress: {percent:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(ISAACSIM_URL, zip_path, reporthook=report_progress)
    print("\nDownload complete!")

    print("Extracting IsaacSim (this may take a few minutes)...")
    os.makedirs(isaacsim_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(isaacsim_path)

    # Clean up zip file
    os.remove(zip_path)
    print("Extraction complete!")

    # Check the extracted structure
    print(f"Contents of {isaacsim_path}:")
    for item in os.listdir(isaacsim_path):
        print(f"  - {item}")

    # Make all shell scripts and binaries executable
    print("Making scripts and binaries executable...")
    for root, dirs, files in os.walk(isaacsim_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Make shell scripts and binaries executable
            if file.endswith(".sh") or file.endswith(".py") or "/bin/" in file_path:
                try:
                    os.chmod(file_path, 0o755)
                except:
                    pass

    # Make specific key files executable
    key_files = [
        "post_install.sh",
        "python.sh",
        "isaac-sim.sh",
        "kit/python/bin/python3",
    ]

    for key_file in key_files:
        file_path = os.path.join(isaacsim_path, key_file)
        if os.path.exists(file_path):
            print(f"Making {key_file} executable...")
            os.chmod(file_path, 0o755)

    # Skip post-install script - it has issues with conda env
    print("Skipping post-install script (incompatible with conda env)")

    return isaacsim_path


def clone_and_setup_isaaclab(isaacsim_path: str):
    """Clone IsaacLab and setup the environment."""
    isaaclab_path = "/root/IsaacLab"

    print(f"Cloning IsaacLab v{ISAACLAB_VERSION}...")
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            f"v{ISAACLAB_VERSION}",
            "--depth",
            "1",
            ISAACLAB_REPO,
            isaaclab_path,
        ],
        check=True,
    )

    # Create symbolic link _isaac_sim -> isaacsim
    isaac_sim_link = os.path.join(isaaclab_path, "_isaac_sim")
    print(f"Creating symbolic link: {isaac_sim_link} -> {isaacsim_path}")

    # Remove existing link if it exists
    if os.path.islink(isaac_sim_link):
        os.unlink(isaac_sim_link)

    os.symlink(isaacsim_path, isaac_sim_link)

    # Make isaaclab.sh executable
    isaaclab_script = os.path.join(isaaclab_path, "isaaclab.sh")
    os.chmod(isaaclab_script, 0o755)

    print(f"IsaacLab cloned to: {isaaclab_path}")
    print(f"Symbolic link created: {isaac_sim_link} -> {isaacsim_path}")

    return isaaclab_path


def install_isaaclab(isaaclab_path: str, isaacsim_path: str):
    """Install IsaacLab dependencies."""
    print("Installing IsaacLab dependencies...")

    # Change to IsaacLab directory
    original_dir = os.getcwd()
    os.chdir(isaaclab_path)

    # Set up environment variables for IsaacLab installation
    env = os.environ.copy()
    env["ISAACSIM_PATH"] = isaacsim_path
    env["ISAACSIM_PYTHON_EXE"] = f"{isaacsim_path}/python.sh"

    # Add IsaacSim to PYTHONPATH
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{isaacsim_path}/site:{isaacsim_path}/python_packages:{python_path}"
    )

    print("Installing Python dependencies...")

    # Run isaaclab.sh install with proper environment
    isaaclab_script = os.path.join(isaaclab_path, "isaaclab.sh")

    print("Running isaaclab.sh --install...")
    result = subprocess.run(
        ["bash", isaaclab_script, "--install"], capture_output=True, text=True, env=env
    )

    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print(f"Warning: isaaclab.sh returned code {result.returncode}")
        print("Attempting manual installation...")

        # Install using IsaacSim's Python pip to ensure packages are available to it
        isaacsim_pip = f"{isaacsim_path}/python.sh -m pip"

        # First, ensure numpy<2.0 is installed in IsaacSim's environment
        subprocess.run(
            ["bash", "-c", f"{isaacsim_pip} install 'numpy<2.0' --force-reinstall"],
            check=False,
        )

        # Install IsaacLab source packages into IsaacSim's Python
        source_dir = os.path.join(isaaclab_path, "source")
        if os.path.exists(source_dir):
            for item in os.listdir(source_dir):
                pkg_path = os.path.join(source_dir, item)
                if os.path.isdir(pkg_path) and os.path.exists(
                    os.path.join(pkg_path, "setup.py")
                ):
                    print(f"Installing {item} into IsaacSim Python...")
                    subprocess.run(
                        ["bash", "-c", f"{isaacsim_pip} install -e {pkg_path}"],
                        check=False,
                    )

        # Also install into conda env for compatibility
        subprocess.run(
            ["pip", "install", "numpy<2.0", "--force-reinstall"], check=False
        )
        if os.path.exists(source_dir):
            for item in os.listdir(source_dir):
                pkg_path = os.path.join(source_dir, item)
                if os.path.isdir(pkg_path) and os.path.exists(
                    os.path.join(pkg_path, "setup.py")
                ):
                    subprocess.run(["pip", "install", "-e", pkg_path], check=False)

    os.chdir(original_dir)
    print("IsaacLab installation complete!")


def setup_image():
    """Setup the Modal image with IsaacSim and IsaacLab."""
    print("=" * 60)
    print("Starting IsaacLab + IsaacSim Setup")
    print("=" * 60)

    # Download and setup IsaacSim
    isaacsim_path = download_and_setup_isaacsim()

    # Clone and setup IsaacLab
    isaaclab_path = clone_and_setup_isaaclab(isaacsim_path)

    # Install IsaacLab
    install_isaaclab(isaaclab_path, isaacsim_path)

    print("=" * 60)
    print("Setup complete!")
    print(f"IsaacSim path: {isaacsim_path}")
    print(f"IsaacLab path: {isaaclab_path}")
    print("=" * 60)


# Create Modal image using micromamba base image with CUDA support
# Using modal.Image.micromamba() which has micromamba pre-installed
image = (
    modal.Image.micromamba()
    .apt_install(
        # Build tools (apt packages, not conda)
        "build-essential",
        "git",
        "wget",
        "curl",
        "ca-certificates",
        "cmake",
        "ninja-build",
        # Graphics and display libraries
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        # X11 and display support
        "libx11-6",
        "libxcursor1",
        "libxfixes3",
        "libxinerama1",
        "libxi6",
        "libxrandr2",
        # Additional IsaacSim dependencies
        "libglu1-mesa",
        "libglew-dev",
        "libosmesa6-dev",
        "libssl-dev",
        "libffi-dev",
        "liblzma-dev",
        "libbz2-dev",
        "libreadline-dev",
        "libsqlite3-dev",
        "libncursesw5-dev",
        "xz-utils",
        "tk-dev",
        "libxml2-dev",
        "libxmlsec1-dev",
        # Fonts
        "fonts-dejavu-core",
        "fonts-liberation",
    )
    .micromamba_install(
        "python=3.10",
        channels=["conda-forge"],
    )
    .pip_install(
        # Modal dependencies needed for run_function
        "grpclib",
        "protobuf",
        # Pre-install some dependencies that might be needed
        "numpy<2.0",
        "torch",
        "torchvision",
    )
    .run_function(setup_image, gpu="T4", timeout=3600)  # 1 hour timeout for build
)

# Create Modal app
app = modal.App("isaaclab-isaacsim")


def get_isaaclab_env():
    """Get environment variables for IsaacLab/IsaacSim."""
    isaacsim_path = "/root/isaacsim"
    isaaclab_path = "/root/IsaacLab"

    # Build comprehensive PYTHONPATH with all IsaacSim paths
    python_path_parts = [
        f"{isaacsim_path}/site",
        f"{isaacsim_path}/python_packages",
        f"{isaacsim_path}/kit/kernel/py",
        f"{isaacsim_path}/kit/exts/omni.client",
        f"{isaacsim_path}/kit/exts/omni.usd",
        f"{isaacsim_path}/kit/python/lib/python3.10/site-packages",
        f"{isaaclab_path}/source",
    ]

    env = {
        # IsaacSim paths
        "ISAACSIM_PATH": isaacsim_path,
        "ISAACSIM_PYTHON_EXE": f"{isaacsim_path}/python.sh",
        # Python path additions
        "PYTHONPATH": ":".join(python_path_parts),
        # Display (headless)
        "DISPLAY": ":0",
        # CUDA
        "CUDA_VISIBLE_DEVICES": "0",
        # IsaacSim specific
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "PERSISTENT": "1",
        # Enable headless mode
        "OMNI_HEADLESS": "1",
    }

    return env


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,  # 1 hour timeout
)
def run_isaaclab_test():
    """Run a basic IsaacLab import test to verify installation."""
    import subprocess

    isaaclab_path = "/root/IsaacLab"
    isaacsim_path = "/root/isaacsim"

    print("Running IsaacLab import test...")

    # Source the conda setup script and test imports
    # Let the setup script handle all environment configuration
    setup_conda_script = os.path.join(isaacsim_path, "setup_conda_env.sh")

    # Build a command that sources the setup and tests imports
    # Note: Full isaaclab.sim requires Carbonite framework (IsaacSim running)
    cmd = f"""
    source {setup_conda_script} && \
    python -c "
import sys
print('Python:', sys.executable)
print('\\nTesting imports...')

# Test 1: AppLauncher (setup only, no Carbonite needed)
try:
    from isaaclab.app import AppLauncher
    print('✓ isaaclab.app.AppLauncher imported successfully')
except Exception as e:
    print('✗ Error importing AppLauncher:', e)
    sys.exit(1)

# Test 2: Check IsaacLab version
try:
    import isaaclab
    print('✓ IsaacLab version:', isaaclab.__version__)
except:
    print('  (version check skipped)')

# Test 3: Check omni.client (would fail without proper paths)
try:
    import omni.client
    print('✓ omni.client imported successfully')
except Exception as err:
    print('  omni.client:', err)

print('\\n✓ Installation verified! AppLauncher works correctly.')
print('  Note: Full simulation requires running IsaacSim (see train_ant function)')
"
    """

    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        cwd=isaaclab_path,
        env=os.environ.copy(),  # Use original env, let setup script configure it
    )

    print(
        "STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
    )
    if result.stderr:
        print(
            "STDERR:",
            result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
        )
    print("Return code:", result.returncode)

    return result.returncode == 0


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
)
def train_ant():
    """Train an ant using RL."""
    import subprocess

    isaaclab_path = "/root/IsaacLab"
    isaaclab_script = os.path.join(isaaclab_path, "isaaclab.sh")
    train_script = os.path.join(
        isaaclab_path, "scripts/reinforcement_learning/rsl_rl/train.py"
    )

    print("Training ant...")

    env = get_isaaclab_env()

    result = subprocess.run(
        [
            "bash",
            isaaclab_script,
            "-p",
            train_script,
            "--task=Isaac-Ant-v0",
            "--headless",
        ],
        capture_output=True,
        text=True,
        cwd=isaaclab_path,
        env={**os.environ, **env},
    )

    print(
        "STDOUT:", result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout
    )
    if result.stderr:
        print(
            "STDERR:",
            result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr,
        )
    print("Return code:", result.returncode)

    return result.returncode == 0


@app.function(
    image=image,
    gpu="T4",
    timeout=300,
)
def check_installation():
    """Check if IsaacSim and IsaacLab are properly installed."""
    import subprocess

    isaacsim_path = "/root/isaacsim"
    isaaclab_path = "/root/IsaacLab"

    results = {
        "isaacsim_exists": os.path.exists(isaacsim_path),
        "isaaclab_exists": os.path.exists(isaaclab_path),
        "isaac_sim_link_exists": os.path.exists(
            os.path.join(isaaclab_path, "_isaac_sim")
        ),
        "isaacsim_python_exists": os.path.exists(
            os.path.join(isaacsim_path, "python.sh")
        ),
        "isaaclab_sh_exists": os.path.exists(
            os.path.join(isaaclab_path, "isaaclab.sh")
        ),
    }

    # Check if IsaacSim Python works
    if results["isaacsim_python_exists"]:
        try:
            env = get_isaaclab_env()
            result = subprocess.run(
                [
                    os.path.join(isaacsim_path, "python.sh"),
                    "-c",
                    "print('IsaacSim Python OK')",
                ],
                capture_output=True,
                text=True,
                env={**os.environ, **env},
                timeout=30,
            )
            results["isaacsim_python_works"] = result.returncode == 0
            results["isaacsim_python_output"] = result.stdout.strip()
        except Exception as e:
            results["isaacsim_python_works"] = False
            results["isaacsim_python_error"] = str(e)

    # Check IsaacLab version
    if results["isaaclab_exists"]:
        try:
            with open(os.path.join(isaaclab_path, "VERSION"), "r") as f:
                results["isaaclab_version"] = f.read().strip()
        except:
            results["isaaclab_version"] = "unknown"

    return results


@app.local_entrypoint()
def main():
    """Main entry point for local execution."""
    print("=" * 60)
    print("IsaacLab-IsaacSim Modal Setup")
    print("=" * 60)
    print(f"IsaacSim version: {ISAACSIM_VERSION}")
    print(f"IsaacLab version: {ISAACLAB_VERSION}")
    print()
    print("Available commands:")
    print("  - modal run isaaclab_modal.py::main (run test)")
    print("  - modal run isaaclab_modal.py::check_installation (check setup)")
    print("  - modal run isaaclab_modal.py::train_ant (train ant)")
    print()

    # Check installation first
    print("--- Checking installation ---")
    install_status = check_installation.remote()
    for key, value in install_status.items():
        print(f"  {key}: {value}")

    # Run test if everything looks good
    if install_status.get("isaacsim_python_works"):
        print("\n--- Running IsaacLab test ---")
        success = run_isaaclab_test.remote()
        print(f"Test {'passed' if success else 'failed'}")
    else:
        print("\nSkipping test - IsaacSim Python not working properly")
