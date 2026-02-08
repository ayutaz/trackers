"""YOLO + ByteTrack demo: detect and track objects in a video file.

Usage::

    python examples/demo_bytetrack.py --source path/to/video.mp4

Requirements::

    uv add ultralytics opencv-python
"""

import argparse
import time

import cv2
import supervision as sv
from ultralytics import YOLO

from trackers import ByteTrackTracker


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO + ByteTrack object tracking demo"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Path for the output video (default: output.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m.pt",
        help="YOLO model name or path (default: yolo11m.pt)",
    )
    return parser.parse_args()


def main() -> None:
    """Run YOLO detection + ByteTrack tracking on a video."""
    args = parse_args()

    # Load YOLO model
    model = YOLO(args.model)

    # Initialize tracker
    tracker = ByteTrackTracker()

    # Set up annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Open input video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: cannot open video '{args.source}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Open output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    prev_time = time.time()

    print(f"Processing '{args.source}' ({total_frames} frames, {fps:.1f} FPS)")
    print(f"Output: '{args.output}'")
    print("Press 'q' to stop early.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects with YOLO
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Track objects with ByteTrack
        detections = tracker.update(detections)

        # Build labels: "#{id}" for matched, "?" for unmatched
        tracker_ids = (
            detections.tracker_id
            if detections.tracker_id is not None
            else []
        )
        labels = [
            f"#{int(tid)}" if tid >= 0 else "?"
            for tid in tracker_ids
        ]

        # Annotate frame
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        # Calculate and display processing FPS
        now = time.time()
        processing_fps = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        frame_idx += 1
        n_tracks = (
            int((detections.tracker_id >= 0).sum())
            if detections.tracker_id is not None
            else 0
        )
        print(
            f"\rFrame {frame_idx}/{total_frames}"
            f"  FPS: {processing_fps:.1f}"
            f"  Tracks: {n_tracks}",
            end="",
            flush=True,
        )

        writer.write(annotated)

        # Show in window; press 'q' to quit early
        cv2.imshow("ByteTrack Demo", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nStopped by user.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n\nDone. Output saved to '{args.output}'")


if __name__ == "__main__":
    main()
