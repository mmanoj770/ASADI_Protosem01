import os
from typing import Tuple
from scenedetect import open_video, SceneManager, ContentDetector

def adjust_clip_to_scene_boundaries(
    video_path: str,
    target_start_sec: float,
    target_end_sec: float,
    search_window: float = 2.0
) -> Tuple[float, float]:
    """
    Adjusts the clip's start and end times to the nearest visual scene cuts,
    ensuring we don't start or end mid-camera-shot.
    Only adjusts if a scene cut is found within the `search_window`.
    """
    if not os.path.exists(video_path):
        return target_start_sec, target_end_sec

    adjusted_start = target_start_sec
    adjusted_end = target_end_sec

    try:
        def find_closest_cut(target_sec: float) -> float:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector())
            
            # Seek to search window start
            start_tc = max(0.0, target_sec - search_window)
            video.seek(start_tc)
            
            # Detect scenes for a short duration
            duration = search_window * 2
            scene_manager.detect_scenes(video, duration=duration)
            scene_list = scene_manager.get_scene_list()
            
            closest_cut = target_sec
            min_diff = float("inf")
            
            for scene in scene_list:
                cut_sec = scene[0].get_seconds()
                diff = abs(cut_sec - target_sec)
                if diff < min_diff and diff <= search_window:
                    min_diff = diff
                    closest_cut = cut_sec
                    
            return closest_cut

        adjusted_start = find_closest_cut(target_start_sec)
        adjusted_end = find_closest_cut(target_end_sec)

        # Ensure we don't accidentally invert or make the clip too short
        if adjusted_end - adjusted_start < 5.0:
            return target_start_sec, target_end_sec

    except Exception as e:
        print(f"[Warning] PySceneDetect failed to adjust boundaries: {e}")
        return target_start_sec, target_end_sec

    return adjusted_start, adjusted_end
