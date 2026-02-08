---
comments: true
---

# Demo: YOLO + ByteTrack

A ready-to-run script that detects objects with [Ultralytics YOLO](https://docs.ultralytics.com/) and tracks them with [`ByteTrackTracker`](../trackers/bytetrack.md). The annotated result is displayed on screen and saved as a video file.

## Result

| Input | Output (with tracking) |
| :---: | :--------------------: |
| ![input](../assets/people-walking.gif) | ![output](../assets/output-bytetrack-demo.gif) |

## Requirements

Install the extra dependencies (trackers and supervision are already included):

=== "pip"

    ```bash
    pip install ultralytics opencv-python
    ```

=== "uv"

    ```bash
    uv add ultralytics opencv-python
    ```

## Usage

```bash
python examples/demo_bytetrack.py --source path/to/video.mp4
```

| Argument | Default | Description |
| :------: | :-----: | :---------: |
| `--source` | *(required)* | Path to input video file |
| `--output` | `output.mp4` | Path for the output video |
| `--model` | `yolo11m.pt` | YOLO model name or path |

Press ++q++ to stop early.

## Quick test with a sample video

You can download a sample video from the [supervision](https://supervision.roboflow.com/) library:

```python
from supervision.assets import download_assets, VideoAssets

video_path = download_assets(VideoAssets.PEOPLE_WALKING)
```

Then run:

```bash
python examples/demo_bytetrack.py --source people-walking.mp4
```

## Source code

The full script is at [`examples/demo_bytetrack.py`](https://github.com/roboflow/trackers/blob/develop/examples/demo_bytetrack.py).

```python
import argparse
import time

import cv2
import supervision as sv
from ultralytics import YOLO

from trackers import ByteTrackTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--model", type=str, default="yolo11m.pt")
    args = parser.parse_args()

    model = YOLO(args.model)
    tracker = ByteTrackTracker()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update(detections)

        tracker_ids = (
            detections.tracker_id
            if detections.tracker_id is not None
            else []
        )
        labels = [
            f"#{int(tid)}" if tid >= 0 else "?"
            for tid in tracker_ids
        ]

        annotated = box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        writer.write(annotated)
        cv2.imshow("ByteTrack Demo", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```
