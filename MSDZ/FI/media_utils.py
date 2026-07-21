import os

import cv2
from PIL import Image


def create_sequence_media(
    frame_dir,
    num_frames,
    fps=15.0,
    save_video=True,
    save_gif=True,
    video_dir=None,
    gif_dir=None,
    output_name="interpolation",
):
    """Create MP4 and GIF files from zero-padded PNG frames in frame_dir."""
    if not save_video and not save_gif:
        return []
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    frame_paths = [
        os.path.join(frame_dir, "{:03d}.png".format(index))
        for index in range(num_frames)
    ]
    missing_paths = [path for path in frame_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(
            "Cannot create media because frame is missing: {}".format(missing_paths[0])
        )

    frames = []
    frame_size = None
    for path in frame_paths:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to read frame: {}".format(path))

        height, width = frame.shape[:2]
        current_size = (width, height)
        if frame_size is None:
            frame_size = current_size
        elif current_size != frame_size:
            raise ValueError(
                "All frames must have the same resolution. {} is {}, expected {}".format(
                    path, current_size, frame_size
                )
            )
        frames.append(frame)

    output_paths = []
    if save_video:
        video_dir = frame_dir if video_dir is None else video_dir
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, "{}.mp4".format(output_name))
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            frame_size,
        )
        if not writer.isOpened():
            raise RuntimeError(
                "Failed to create MP4 with the OpenCV mp4v codec: {}".format(video_path)
            )
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
            raise RuntimeError("OpenCV created an empty MP4 file: {}".format(video_path))
        output_paths.append(video_path)

    if save_gif:
        gif_dir = frame_dir if gif_dir is None else gif_dir
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, "{}.gif".format(output_name))
        gif_frames = [
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for frame in frames
        ]
        frame_duration_ms = max(1, int(round(1000.0 / fps)))
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )
        output_paths.append(gif_path)

    print("Created media: {}".format(", ".join(output_paths)))
    return output_paths
