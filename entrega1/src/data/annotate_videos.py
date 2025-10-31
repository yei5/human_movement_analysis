import cv2
import json
import os

def generate_annotation(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    base = os.path.basename(video_path)
    activity = base.split("-")[0] if "-" in base else os.path.splitext(base)[0]

    annotation = [
        {
            "activity": activity,
            "start_frame": 0,
            "start_time": 0.0,
            "end_frame": frame_count - 1,
            "end_time": duration,
            "method": "automatic_filename"
        }
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotation, f, indent=2)

    print(f"Annotated {video_path} → {output_path}")

if __name__ == "__main__":
    input_dir = "data/raw_videos/"
    output_dir = "data/processed/annotations/"
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if fname.endswith(".mp4"):
            video_path = os.path.join(input_dir, fname)
            ann_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.json")
            generate_annotation(video_path, ann_path)
