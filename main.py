import argparse
import sys
from src.camera import GestureAppCamera

def main():
    """
    Main entry point for the hand gesture recognition application.
    Parses command-line arguments and starts the camera application.
    """
    parser = argparse.ArgumentParser(
        description="Real-time Python Hand Gesture Recognition App using OpenCV & MediaPipe."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Index of the camera device to use (default: 0)."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width of the webcam capture frame (default: 1280)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height of the webcam capture frame (default: 720)."
    )
    
    args = parser.parse_args()

    print("====================================================")
    print("      HAND GESTURE RECOGNITION APPLICATION")
    print("====================================================")
    print(f"Camera Device Index : {args.camera}")
    print(f"Resolution          : {args.width}x{args.height}")
    print("Press 'q' in the window to quit at any time.")
    print("====================================================")

    try:
        app = GestureAppCamera(
            camera_index=args.camera,
            width=args.width,
            height=args.height
        )
        app.run()
    except Exception as e:
        print(f"Critical error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
