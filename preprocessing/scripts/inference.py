"""
Real-time sign language inference from webcam or video file
"""
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import sys
import mediapipe as mp

# Add parent directory to path (go up from scripts/ to preprocessing/)
sys.path.append(str(Path(__file__).parent.parent))

from config import SEQUENCE_LENGTH, CHECKPOINT_DIR
from utils.keypoint_extraction import extract_keypoints

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def load_model_and_actions(model_path=None, mapping_path=None):
    """Load trained model and action mapping"""
    if model_path is None:
        model_path = CHECKPOINT_DIR / 'best_model.h5'
    if mapping_path is None:
        mapping_path = CHECKPOINT_DIR / 'action_mapping.json'
    
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading action mapping: {mapping_path}")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        action_mapping = json.load(f)
    
    # Convert string keys to int
    action_mapping = {int(k): v for k, v in action_mapping.items()}
    
    return model, action_mapping


def inference_webcam(model, action_mapping, threshold=0.7):
    """
    Real-time inference from webcam
    
    Args:
        model: Trained model
        action_mapping: Dict mapping class indices to action names
        threshold: Confidence threshold for prediction
    """
    # Setup webcam
    cap = cv2.VideoCapture(0)
    
    # Detection variables
    sequence = []
    predictions = []
    current_action = "Waiting..."
    confidence = 0.0
    
    print("\n" + "="*70)
    print("REAL-TIME SIGN LANGUAGE INFERENCE")
    print("="*70)
    print(f"Model: {len(action_mapping)} actions")
    print(f"Sequence length: {SEQUENCE_LENGTH} frames")
    print(f"Confidence threshold: {threshold}")
    print("\nPress 'q' to quit")
    print("="*70 + "\n")
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for selfie view
            frame = cv2.flip(frame, 1)
            
            # MediaPipe detection
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 200), thickness=2)
            )
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1)
            )
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2)
            )
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(200, 0, 200), thickness=2)
            )
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]  # Keep last N frames
            
            # Predict when we have enough frames
            if len(sequence) == SEQUENCE_LENGTH:
                # Reshape for model: (1, 130, 1662)
                input_data = np.expand_dims(sequence, axis=0)
                
                # Predict
                pred = model.predict(input_data, verbose=0)[0]
                predicted_class = np.argmax(pred)
                confidence = pred[predicted_class]
                
                # Update prediction if confidence is high
                if confidence > threshold:
                    current_action = action_mapping[predicted_class]
                else:
                    current_action = "Uncertain..."
                
                # Store prediction history
                predictions.append(predicted_class)
                predictions = predictions[-10:]  # Keep last 10 predictions
            
            # Display UI at BOTTOM (avoid covering original video text)
            h, w, _ = image.shape
            
            # Background for text at bottom
            cv2.rectangle(image, (0, h-120), (w, h), (0, 0, 0), -1)
            
            # Current prediction
            color = (0, 255, 0) if confidence > threshold else (0, 165, 255)
            cv2.putText(image, f"Prediction: {current_action}", 
                       (10, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
            # Confidence
            cv2.putText(image, f"Confidence: {confidence*100:.1f}%", 
                       (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Sequence progress bar
            progress = len(sequence) / SEQUENCE_LENGTH
            cv2.rectangle(image, (10, h-20), (10 + int(progress * (w-20)), h-10), 
                         (0, 255, 0), -1)
            
            # Show frame
            cv2.imshow('Sign Language Recognition - Press Q to quit', image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nInference stopped.")


def inference_video(model, action_mapping, video_path, threshold=0.7, output_path=None, headless=False):
    """
    Inference from video file
    
    Args:
        model: Trained model
        action_mapping: Dict mapping class indices to action names
        video_path: Path to input video
        threshold: Confidence threshold
        output_path: Path to save output video (optional)
        headless: If True, don't show GUI (for servers without display)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Video writer setup
    if output_path:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Detection variables
    sequence = []
    current_action = "Waiting..."
    confidence = 0.0
    
    print(f"\nProcessing video: {video_path}")
    if output_path:
        print(f"Output will be saved to: {output_path}")
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # MediaPipe detection
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks (same as webcam)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]
            
            # Predict
            if len(sequence) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(sequence, axis=0)
                pred = model.predict(input_data, verbose=0)[0]
                predicted_class = np.argmax(pred)
                confidence = pred[predicted_class]
                
                if confidence > threshold:
                    current_action = action_mapping[predicted_class]
                else:
                    current_action = "Uncertain..."
            
            # Display UI at BOTTOM
            h, w, _ = image.shape
            cv2.rectangle(image, (0, h-80), (w, h), (0, 0, 0), -1)
            color = (0, 255, 0) if confidence > threshold else (0, 165, 255)
            cv2.putText(image, f"{current_action} ({confidence*100:.1f}%)", 
                       (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
            # Show frame (only if not headless)
            if not headless:
                cv2.imshow('Processing - Press Q to stop', image)
            
            if output_path:
                out.write(image)
            
            if not headless and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    if output_path:
        out.release()
        print(f"\n✓ Output video saved: {output_path}")
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sign Language Recognition Inference')
    parser.add_argument('--mode', type=str, default='webcam', choices=['webcam', 'video'],
                       help='Inference mode: webcam or video')
    parser.add_argument('--video', type=str, help='Path to input video (for video mode)')
    parser.add_argument('--output', type=str, help='Path to output video (optional)')
    parser.add_argument('--model', type=str, help='Path to model file (default: checkpoints/best_model.h5)')
    parser.add_argument('--threshold', type=float, default=0.7, 
                       help='Confidence threshold (default: 0.7)')
    parser.add_argument('--headless', action='store_true',
                       help='Run without GUI (for servers without display)')
    
    args = parser.parse_args()
    
    # Load model
    model, action_mapping = load_model_and_actions(
        model_path=args.model if args.model else None
    )
    
    # Run inference
    if args.mode == 'webcam':
        inference_webcam(model, action_mapping, threshold=args.threshold)
    else:
        if not args.video:
            print("Error: --video path required for video mode")
            sys.exit(1)
        inference_video(model, action_mapping, args.video, 
                       threshold=args.threshold, output_path=args.output, headless=args.headless)
