"""
SatDump IQ recording processing task.

This task processes IQ recordings using SatDump with various satellite pipelines.
Supports progress tracking and graceful interruption.
"""

import os
import re
import shutil
import signal
import subprocess
from multiprocessing import Queue
from pathlib import Path
from typing import Optional

from PIL import Image


class GracefulKiller:
    """Handle SIGTERM gracefully within the process."""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        signal.signal(signal.SIGINT, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


def _cleanup_empty_directory(directory: Path, progress_queue: Optional[Queue] = None):
    """
    Remove directory if it's empty or contains only empty subdirectories.

    Args:
        directory: Path to check and potentially remove
        progress_queue: Optional queue for logging
    """
    try:
        if not directory.exists():
            return

        # Check if directory has any files (recursively)
        has_files = any(directory.rglob("*"))

        if not has_files or _is_directory_empty(directory):
            if progress_queue:
                progress_queue.put(
                    {
                        "type": "output",
                        "output": f"Cleaning up empty output directory: {directory}",
                        "stream": "stdout",
                    }
                )
            shutil.rmtree(directory)
    except Exception as e:
        if progress_queue:
            progress_queue.put(
                {
                    "type": "output",
                    "output": f"Warning: Failed to clean up directory: {e}",
                    "stream": "stderr",
                }
            )


def _is_directory_empty(directory: Path) -> bool:
    """
    Check if directory is empty or contains only empty subdirectories.

    Args:
        directory: Path to check

    Returns:
        True if directory is empty or contains only empty subdirectories
    """
    if not directory.is_dir():
        return False

    for item in directory.iterdir():
        if item.is_file():
            return False
        if item.is_dir() and not _is_directory_empty(item):
            return False

    return True


def _pick_map_thumbnail_source(output_path: Path) -> Optional[Path]:
    """
    Pick the best map image from SatDump output to generate a card thumbnail.
    """
    pngs = list(output_path.rglob("*.png"))
    if not pngs:
        return None

    def score(path: Path) -> int:
        name = path.name.lower()
        full = str(path).lower()
        value = 0
        # Best visual for METEOR in this pipeline: projected false-color overlay.
        if name == "rgb_msu_mr_rgb_avhrr_3a21_false_color_projected.png":
            value += 300
        if "_projected" in name:
            value += 180
        if name.endswith("_map.png"):
            value += 120
        if "corrected_map" in full:
            value -= 10
        if "rgb" in full:
            value += 30
        return value

    return max(pngs, key=lambda p: (score(p), str(p)))


def _generate_decoded_thumbnail(
    output_path: Path, progress_queue: Optional[Queue] = None
) -> Optional[Path]:
    """
    Generate a stable decoded-folder thumbnail file at output root.
    """
    source = _pick_map_thumbnail_source(output_path)
    if not source:
        return None

    thumb_path = output_path / "thumbnail.jpg"

    try:
        with Image.open(source) as img:
            # 16:9 target fits file-browser cards.
            # Preserve full map content (no center-crop), add subtle letterbox padding if needed.
            target_w, target_h = 960, 540
            rgb = img.convert("RGB")
            rgb.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (target_w, target_h), color=(10, 10, 10))
            x = (target_w - rgb.width) // 2
            y = (target_h - rgb.height) // 2
            canvas.paste(rgb, (x, y))
            canvas.save(thumb_path, format="JPEG", quality=88, optimize=True)

        if progress_queue:
            progress_queue.put(
                {
                    "type": "output",
                    "output": f"Generated folder thumbnail from: {source.name}",
                    "stream": "stdout",
                }
            )
        return thumb_path
    except Exception as e:
        if progress_queue:
            progress_queue.put(
                {
                    "type": "output",
                    "output": f"Warning: thumbnail generation failed: {e}",
                    "stream": "stderr",
                }
            )
        return None


def satdump_process_recording(
    recording_path: str,
    output_dir: str,
    satellite: str,
    samplerate: int = 0,
    baseband_format: str = "i16",
    finish_processing: bool = True,
    delete_input_after: bool = False,
    _progress_queue: Optional[Queue] = None,
):
    """
    Process an IQ recording with SatDump.

    Args:
        recording_path: Path to the input IQ recording file
        output_dir: Output directory for decoded products
        satellite: Satellite/pipeline identifier (e.g., 'meteor_m2-x_lrpt', 'noaa_apt')
        samplerate: Sample rate in Hz (0 = auto-detect from filename)
        baseband_format: Input format ('i16', 'i8', 'f32', 'w16', 'w8', etc.)
        finish_processing: Whether to run product generation after decoding
        _progress_queue: Queue for sending progress updates

    Returns:
        Dict with processing results
    """
    killer = GracefulKiller()

    # Resolve paths relative to backend/data directory
    backend_dir = Path(__file__).parent.parent

    recording_file = Path(recording_path)
    # Check if path starts with /recordings/ or /decoded/ (relative app paths)
    if not recording_file.exists():
        recording_file = backend_dir / "data" / recording_path.lstrip("/")

    output_path = Path(output_dir)
    if not output_path.is_absolute() or str(output_path).startswith("/decoded"):
        output_path = backend_dir / "data" / output_dir.lstrip("/")

    # Validate inputs
    if not recording_file.exists():
        error_msg = f"Recording file not found: {recording_file}"
        if _progress_queue:
            _progress_queue.put({"type": "error", "error": error_msg, "stream": "stderr"})
        raise FileNotFoundError(error_msg)

    # Create output directory
    output_dir_preexisted = output_path.exists()
    output_path.mkdir(parents=True, exist_ok=True)

    # Build SatDump command using resolved absolute paths (SatDump 1.2.x CLI)
    # satdump <pipeline> baseband <in> <out> --samplerate <sr> --baseband_format <fmt> ...
    fmt_map = {
        "f32": "cf32",
        "i16": "cs16",
        "i8": "cs8",
        "u8": "cu8",
        "w16": "w16",
        "w8": "w8",
    }
    baseband_format = fmt_map.get(baseband_format, baseband_format)

    cmd = [
        "satdump",
        satellite,
        "baseband",
        str(recording_file),
        str(output_path),
        "--samplerate",
        str(samplerate),
        "--baseband_format",
        str(baseband_format),
        "--fill_missing",
        "--dc_block",
    ]

    # Log command
    if _progress_queue:
        _progress_queue.put(
            {
                "type": "output",
                "output": "Starting SatDump processing",
                "stream": "stdout",
            }
        )
        _progress_queue.put(
            {
                "type": "output",
                "output": f"Satellite/Pipeline: {satellite}",
                "stream": "stdout",
            }
        )
        _progress_queue.put(
            {
                "type": "output",
                "output": f"Input: {recording_file}",
                "stream": "stdout",
            }
        )
        _progress_queue.put(
            {
                "type": "output",
                "output": f"Output: {output_path}",
                "stream": "stdout",
            }
        )
        _progress_queue.put(
            {"type": "output", "output": f"Command: {' '.join(cmd)}", "stream": "stdout"}
        )
        _progress_queue.put({"type": "output", "output": "-" * 60, "stream": "stdout"})

    try:
        # Isolate SatDump settings to avoid cross-version config breakage
        satdump_cfg_dir = backend_dir / "data" / "satdump_config"
        satdump_cfg_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["XDG_CONFIG_HOME"] = str(satdump_cfg_dir)

        # Start the subprocess
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Stream output
        last_progress: Optional[float] = None
        while True:
            # Check for graceful shutdown
            if killer.kill_now:
                if _progress_queue:
                    _progress_queue.put(
                        {
                            "type": "output",
                            "output": "Terminating SatDump process...",
                            "stream": "stdout",
                        }
                    )
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                return {"status": "interrupted", "message": "Process was interrupted"}

            # Read output
            if process.stdout:
                line = process.stdout.readline()
                if not line:
                    # Process finished
                    break
            else:
                break

            line = line.rstrip()
            line = re.sub(r"^\[\d{2}:\d{2}:\d{2}\s+-\s+\d{2}/\d{2}/\d{4}\]\s*", "", line)
            if line and _progress_queue:
                # Filter out debug (D) and trace (T) messages from SatDump
                if "(D)" in line or "(T)" in line:
                    continue

                # Filter out "Loading pipelines from file" messages
                if "Loading pipelines from file" in line:
                    continue

                # Parse progress from SatDump output
                progress = None
                if "Progress" in line or "%" in line:
                    # Try to extract percentage
                    try:
                        # Look for patterns like "Progress: 45.2%" or "[45%]"
                        match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
                        if match:
                            progress = float(match.group(1))
                    except Exception:
                        pass
                if progress is not None:
                    if last_progress is None or progress >= last_progress:
                        last_progress = progress
                progress_to_emit = last_progress

                _progress_queue.put(
                    {
                        "type": "output",
                        "output": line,
                        "stream": "stdout",
                        "progress": progress_to_emit if progress_to_emit is not None else None,
                    }
                )

        # Wait for process to complete
        return_code = process.wait()

        # SatDump 1.2.x doesn't have `satdump process`, so skip post-processing here.

        # Check if output directory has any decoded products (images, data files)
        # SatDump v1.2.3 returns exit code 1 even when decoding succeeded
        has_images = False
        has_output = False
        if output_path.exists():
            # Check for any image files or data products
            has_images = (
                any(output_path.rglob("*.png"))
                or any(output_path.rglob("*.jpg"))
                or any(output_path.rglob("*.jpeg"))
            )
            has_output = has_images or any(output_path.rglob("product.cbor"))

        if delete_input_after:
            try:
                base_path = recording_file
                if base_path.name.endswith(".sigmf-data"):
                    base_path = base_path.with_suffix("")

                data_path = base_path.with_suffix(".sigmf-data")
                meta_path = base_path.with_suffix(".sigmf-meta")

                for path in (data_path, meta_path):
                    if path.exists():
                        path.unlink()

                if _progress_queue:
                    _progress_queue.put(
                        {
                            "type": "output",
                            "output": f"Deleted IQ recording: {data_path} (+.sigmf-meta)",
                            "stream": "stdout",
                        }
                    )
            except Exception as e:
                if _progress_queue:
                    _progress_queue.put(
                        {
                            "type": "output",
                            "output": f"Warning: Failed to delete IQ recording: {e}",
                            "stream": "stderr",
                        }
                    )

        if not has_images:
            if output_path.exists() and not output_dir_preexisted:
                if _progress_queue:
                    _progress_queue.put(
                        {
                            "type": "output",
                            "output": (
                                "No decoded images found; removing SatDump output directory"
                            ),
                            "stream": "stdout",
                        }
                    )
                shutil.rmtree(output_path)

            error_msg = "SatDump completed without decoded images"
            if _progress_queue:
                _progress_queue.put({"type": "error", "error": error_msg, "stream": "stderr"})
            raise RuntimeError(error_msg)

        # SatDump v1.2.x returns exit code 1 even when decoding succeeded
        if return_code == 0 or (return_code == 1 and has_output):
            _generate_decoded_thumbnail(output_path, _progress_queue)
            if _progress_queue:
                _progress_queue.put(
                    {
                        "type": "output",
                        "output": "-" * 60,
                        "stream": "stdout",
                        "progress": 100,
                    }
                )
                if return_code == 1:
                    _progress_queue.put(
                        {
                            "type": "output",
                            "output": "SatDump completed with exit code 1 but products were generated successfully",
                            "stream": "stdout",
                        }
                    )
                else:
                    _progress_queue.put(
                        {
                            "type": "output",
                            "output": "SatDump processing completed successfully!",
                            "stream": "stdout",
                        }
                    )
                _progress_queue.put(
                    {
                        "type": "output",
                        "output": f"Output files saved to: {output_path}",
                        "stream": "stdout",
                    }
                )

            return {
                "status": "completed",
                "output_dir": str(output_path),
                "satellite": satellite,
                "return_code": return_code,
            }
        else:
            # Clean up empty output directory
            _cleanup_empty_directory(output_path, _progress_queue)

            error_msg = f"SatDump process failed with return code {return_code}"
            if _progress_queue:
                _progress_queue.put({"type": "error", "error": error_msg, "stream": "stderr"})
            raise RuntimeError(error_msg)

    except Exception as e:
        # Clean up empty output directory on exception
        _cleanup_empty_directory(output_path, _progress_queue)

        error_msg = f"Error during SatDump processing: {str(e)}"
        if _progress_queue:
            _progress_queue.put({"type": "error", "error": error_msg, "stream": "stderr"})
        raise
