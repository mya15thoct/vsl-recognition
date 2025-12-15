"""
Inference script for Sign Language Action Detection
Supports both webcam and video file input
"""
import cv2
import numpy as np
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model

# Import from modules
import sys
sys.path.append(str(Path(__file__).parent))

from config import MODEL_PATH, SEQUENCE_LENGTH
from utils.keypoint_extraction import mediapipe_detection, extract_keypoints, get_holistic_model
from utils.visualization import draw_styled_landmarks


def load_trained_model(model_path=None):
    """
    Load trained model and action labels
    
    Args:
        model_path: Path to model file (default: best model)
        
    Returns:
        model: Loaded Keras model
        actions: List of action names
    """
    if model_path is None:
        model_path = MODEL_PATH / 'action_model_best.h5'
    
    if not Path(model_path).exists():
        model_path = MODEL_PATH / 'action_model_final.h5'
    
    print(f"Loading model from: {model_path}")
    model = load_model(str(model_path))
    
    # Load action labels
    labels_path = MODEL_PATH / 'actions.npy'
    actions = np.load(str(labels_path))
    print(f"Loaded {len(actions)} action classes")
    
    return model, actions


def predict_from_webcam(model, actions, sequence_length=SEQUENCE_LENGTH):
    """
    Real-time prediction from webcam
    
    Args:
        model: Trained Keras model
        actions: List of action names
        sequence_length: Number of frames per sequence
    """
    print("\nStarting webcam prediction...")
    print("Press 'q' to quit")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize MediaPipe
    holistic = get_holistic_model()
    
    # Storage for sequence
    sequence = []
    predictions = []
    threshold = 0.5
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]
        
        # Make prediction when we have enough frames
        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            
            # Get most common prediction from last 10
            if len(predictions) > 10:
                predictions = predictions[-10:]
            
            # Display prediction
            if np.max(res) > threshold:
                action = actions[np.argmax(res)]
                confidence = np.max(res)
                
                # Draw prediction on frame
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, f'{action} ({confidence:.2f})', (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show frame
        cv2.imshow('Sign Language Detection', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()


def predict_from_video(model, actions, video_path, sequence_length=SEQUENCE_LENGTH):
    """
    Prediction from video file
    
    Args:
        model: Trained Keras model
        actions: List of action names
        video_path: Path to video file
        sequence_length: Number of frames per sequence
    """
    print(f"\nProcessing video: {video_path}")
    
    # Initialize video
    cap = cv2.VideoCapture(str(video_path))
    
    # Initialize MediaPipe
    holistic = get_holistic_model()
    
    # Storage for sequence
    sequence = []
    predictions = []
    threshold = 0.5
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]
        
        # Make prediction
        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            
            if np.max(res) > threshold:
                action = actions[np.argmax(res)]
                confidence = np.max(res)
                
                # Draw prediction
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, f'{action} ({confidence:.2f})', (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                print(f"Prediction: {action} (Confidence: {confidence:.2f})")
        
        # Show frame
        cv2.imshow('Sign Language Detection', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sign Language Action Detection - Inference')
    parser.add_argument('--video', type=str, help='Path to video file (if not specified, uses webcam)')
    parser.add_argument('--model', type=str, help='Path to model file (default: best model)')
    
    args = parser.parse_args()
    
    # Load model
    model, actions = load_trained_model(args.model)
    
    # Run inference
    if args.video:
        predict_from_video(model, actions, args.video)
    else:
        predict_from_webcam(model, actions)
