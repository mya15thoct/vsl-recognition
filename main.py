"""
Main CLI entry point for VSL Recognition project

Usage:
    python main.py data prepare     # Prepare dataset
    python main.py train           # Train model
    python main.py inference       # Run inference
    python main.py visualize       # Visualization tools
"""
import sys
import argparse
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description='VSL Recognition - Sign Language Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py data prepare                    # Prepare dataset
    python main.py train                          # Train model  
    python main.py inference --mode webcam        # Run webcam inference
    python main.py visualize keypoints            # Visualize keypoints
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data command
    data_parser = subparsers.add_parser('data', help='Data preparation')
    data_parser.add_argument('action', choices=['prepare', 'check'], 
                             help='prepare: Run full pipeline | check: Check distribution')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--mode', choices=['webcam', 'video'], 
                                  default='webcam', help='Inference mode')
    inference_parser.add_argument('--video', type=str, help='Path to video file')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualization tools')
    viz_parser.add_argument('target', choices=['keypoints', 'sequences'],
                            help='What to visualize')
    viz_parser.add_argument('--class-name', type=str, help='Class name to visualize')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'data':
        if args.action == 'prepare':
            print("Running data preparation pipeline...")
            from src.data.prepare_pipeline import main as prepare_main
            # Note: prepare_pipeline.py runs on import, so we just import it
            import src.data.prepare_pipeline
        elif args.action == 'check':
            from src.data.check_distribution import check_distribution
            check_distribution()
    
    elif args.command == 'train':
        print("Starting training...")
        from src.training.pipeline import run_full_pipeline
        run_full_pipeline()
    
    elif args.command == 'inference':
        print(f"Starting inference (mode: {args.mode})...")
        # Import and run inference module
        # Note: This would need to be adapted based on actual inference script structure
        print("⚠ Inference command not fully implemented yet")
        print("   Run directly: python -m src.inference.realtime --mode webcam")
    
    elif args.command == 'visualize':
        print(f"Visualizing {args.target}...")
        # Note: This would need to be adapted based on actual visualization scripts
        print("⚠ Visualization command not fully implemented yet")
        print(f"   Run directly: python -m src.visualization.{args.target}")


if __name__ == "__main__":
    main()
