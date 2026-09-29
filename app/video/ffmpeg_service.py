import os
import shutil
import subprocess

def check_ffmpeg_installed() -> tuple[bool, str]:
    """Check if ffmpeg is available on PATH."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False, "FFmpeg binary not found on system PATH. Please install FFmpeg."
    try:
        res = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        first_line = res.stdout.splitlines()[0] if res.stdout else "FFmpeg installed"
        return True, first_line
    except Exception as e:
        return False, f"FFmpeg check failed: {e}"

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe or ffmpeg."""
    if not os.path.exists(video_path):
        return 0.0

    # Try ffprobe first
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        try:
            cmd = [
                ffprobe_path, "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(res.stdout.strip())
        except Exception:
            pass

    # Fallback to parsing ffmpeg stderr output
    try:
        cmd = ["ffmpeg", "-i", video_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        for line in res.stderr.splitlines():
            if "Duration:" in line:
                # Duration: 00:01:23.45, start: 0.000000...
                dur_str = line.split("Duration:")[1].split(",")[0].strip()
                p = [float(x) for x in dur_str.replace(",", ".").split(":")]
                if len(p) == 3:
                    return p[0]*3600 + p[1]*60 + p[2]
    except Exception:
        pass
    return 0.0

def extract_audio(video_path: str, output_audio_path: str) -> str:
    """Extract mono 16kHz WAV audio file for optimal speech transcription."""
    # Fast return if already extracted
    if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 1000:
        return output_audio_path

    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-threads", "0", "-v", "error",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-f", "wav", output_audio_path
    ]
    subprocess.run(cmd, check=True)
    return output_audio_path


def cut_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_clip_path: str,
    vertical_crop: bool = False
) -> str:
    """
    Cut video segment from start_sec to end_sec.
    If vertical_crop is True, crops video to 9:16 portrait ratio (1080x1920).
    Uses PySceneDetect to align cuts to visual boundaries.
    """
    try:
        from app.video.scene_service import adjust_clip_to_scene_boundaries
        start_sec, end_sec = adjust_clip_to_scene_boundaries(video_path, start_sec, end_sec)
    except Exception as e:
        print(f"[Warning] Scene adjustment skipped: {e}")

    os.makedirs(os.path.dirname(output_clip_path), exist_ok=True)
    duration = max(0.5, end_sec - start_sec)
    
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start_sec:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}"
    ]

    if vertical_crop:
        # 9:16 center crop filter
        vf = "crop=ih*9/16:ih,scale=1080:1920"
        cmd.extend(["-vf", vf, "-c:v", "libx264", "-crf", "22", "-c:a", "aac"])
    else:
        # Fast copy codec if original aspect ratio is preferred
        cmd.extend(["-c:v", "libx264", "-crf", "20", "-c:a", "aac"])

    cmd.append(output_clip_path)
    subprocess.run(cmd, check=True)
    return output_clip_path
