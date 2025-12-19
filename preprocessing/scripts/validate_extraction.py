"""
Validate MediaPipe Keypoint Extraction Quality
Run this BEFORE extraction to ensure good data quality
"""
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE, DATA_DIR


class MediaPipeValidator:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE
        )
    
    def validate_video(self, video_path, visualize=False):
        """
        Validate MediaPipe extraction on a video
        
        Args:
            video_path: Path to video file
            visualize: Show keypoints overlay (press 'q' to skip)
        
        Returns:
            dict with quality metrics
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Metrics
        detection_count = {
            'pose': 0,
            'face': 0,
            'left_hand': 0,
            'right_hand': 0
        }
        
        confidence_scores = []
        keypoint_positions = []  # For consistency check
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)
            
            # Check pose
            if results.pose_landmarks:
                detection_count['pose'] += 1
                # Get average confidence (visibility)
                confidences = [lm.visibility for lm in results.pose_landmarks.landmark]
                confidence_scores.append(np.mean(confidences))
                
                # Store positions for consistency check
                positions = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
                keypoint_positions.append(positions)
            
            # Check face
            if results.face_landmarks:
                detection_count['face'] += 1
            
            # Check hands
            if results.left_hand_landmarks:
                detection_count['left_hand'] += 1
            
            if results.right_hand_landmarks:
                detection_count['right_hand'] += 1
            
            # Visualize (optional)
            if visualize:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw pose
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image_bgr, results.pose_landmarks, 
                        self.mp_holistic.POSE_CONNECTIONS
                    )
                
                # Draw face
                if results.face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image_bgr, results.face_landmarks,
                        self.mp_holistic.FACEMESH_CONTOURS
                    )
                
                # Draw hands
                if results.left_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image_bgr, results.left_hand_landmarks,
                        self.mp_holistic.HAND_CONNECTIONS
                    )
                
                if results.right_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image_bgr, results.right_hand_landmarks,
                        self.mp_holistic.HAND_CONNECTIONS
                    )
                
                # Add text info
                cv2.putText(image_bgr, f"Frame: {frame_idx}/{total_frames}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('MediaPipe Validation - Press Q to skip', image_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    visualize = False  # Stop visualization
            
            frame_idx += 1
        
        cap.release()
        if visualize:
            cv2.destroyAllWindows()
        
        # Calculate metrics
        detection_rates = {
            key: (count / total_frames * 100) if total_frames > 0 else 0
            for key, count in detection_count.items()
        }
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        # Consistency check (movement smoothness)
        consistency_score = self._check_consistency(keypoint_positions)
        
        return {
            'total_frames': total_frames,
            'detection_rates': detection_rates,
            'avg_confidence': avg_confidence,
            'consistency_score': consistency_score,
            'is_good_quality': self._evaluate_quality(detection_rates, avg_confidence, consistency_score)
        }
    
    def _check_consistency(self, keypoint_positions):
        """
        Check if keypoints move smoothly between frames
        Low variance = smooth = good
        High variance = jumpy = bad
        """
        if len(keypoint_positions) < 2:
            return 0.0
        
        movements = []
        for i in range(1, len(keypoint_positions)):
            prev_kpts = np.array(keypoint_positions[i-1])
            curr_kpts = np.array(keypoint_positions[i])
            
            # Calculate average movement
            movement = np.mean(np.sqrt(np.sum((curr_kpts - prev_kpts)**2, axis=1)))
            movements.append(movement)
        
        # Low variance = smooth movement = good
        variation = np.std(movements) if movements else 1.0
        
        # Convert to score (0-1, higher is better)
        consistency_score = max(0, 1 - variation * 10)
        
        return consistency_score
    
    def _evaluate_quality(self, detection_rates, avg_confidence, consistency_score):
        """
        Overall quality evaluation
        
        Criteria:
        - Pose detection > 90%
        - Confidence > 0.7
        - Consistency > 0.6
        """
        pose_ok = detection_rates['pose'] > 90
        confidence_ok = avg_confidence > 0.7
        consistency_ok = consistency_score > 0.6
        
        return pose_ok and confidence_ok and consistency_ok
    
    def validate_dataset(self, video_folder, num_samples=10, visualize_first=False):
        """
        Validate a sample of videos from dataset
        
        Args:
            video_folder: Path to folder containing videos
            num_samples: Number of videos to sample and validate
            visualize_first: Show keypoints for first video
        """
        video_files = list(Path(video_folder).glob('*.mp4'))
        
        if not video_files:
            print("[ERROR] No videos found in", video_folder)
            return
        
        # Sample random videos
        sample_videos = np.random.choice(
            video_files, 
            min(num_samples, len(video_files)), 
            replace=False
        )
        
        print("="*60)
        print("MEDIAPIPE EXTRACTION QUALITY VALIDATION")
        print("="*60)
        print(f"\nTotal videos in dataset: {len(video_files)}")
        print(f"Validating {len(sample_videos)} sample videos...\n")
        
        all_results = []
        
        for idx, video_path in enumerate(sample_videos):
            # Visualize first video if requested
            show_viz = visualize_first and idx == 0
            
            result = self.validate_video(video_path, visualize=show_viz)
            all_results.append(result)
            
            # Print result
            quality_icon = "✅" if result['is_good_quality'] else "❌"
            print(f"\n{quality_icon} Video {idx+1}: {video_path.name}")
            print(f"   Total frames: {result['total_frames']}")
            print(f"   Pose detection: {result['detection_rates']['pose']:.1f}%")
            print(f"   Face detection: {result['detection_rates']['face']:.1f}%")
            print(f"   Left hand: {result['detection_rates']['left_hand']:.1f}%")
            print(f"   Right hand: {result['detection_rates']['right_hand']:.1f}%")
            print(f"   Avg confidence: {result['avg_confidence']:.3f}")
            print(f"   Consistency: {result['consistency_score']:.3f}")
        
        # Overall summary
        good_count = sum(1 for r in all_results if r['is_good_quality'])
        good_percentage = good_count / len(all_results) * 100
        
        avg_pose_detection = np.mean([r['detection_rates']['pose'] for r in all_results])
        avg_confidence = np.mean([r['avg_confidence'] for r in all_results])
        avg_consistency = np.mean([r['consistency_score'] for r in all_results])
        
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        print(f"Good quality videos: {good_count}/{len(all_results)} ({good_percentage:.1f}%)")
        print(f"Average pose detection: {avg_pose_detection:.1f}%")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average consistency: {avg_consistency:.3f}")
        
        print("\n" + "="*60)
        if good_percentage >= 80:
            print("✅ EXTRACTION QUALITY IS GOOD")
            print("   Ready for LSTM training!")
        elif good_percentage >= 60:
            print("⚠️  EXTRACTION QUALITY IS ACCEPTABLE")
            print("   You can proceed but consider improving:")
            print("   - Check videos with poor quality")
            print("   - Improve lighting/camera angle")
        else:
            print("❌ EXTRACTION QUALITY NEEDS IMPROVEMENT")
            print("   Recommendations:")
            print("   - Check video quality (resolution, lighting)")
            print("   - Ensure clear view of person")
            print("   - Remove corrupted videos")
            print("   - Consider re-recording poor quality videos")
        print("="*60)
        
        self.holistic.close()
        
        return all_results


def main():
    """
    Main function to run validation
    """
    validator = MediaPipeValidator()
    
    # Validate dataset
    video_folder = DATA_DIR / "videos"
    
    print("\nStarting validation...")
    print("This will check MediaPipe extraction quality on sample videos")
    print("Press 'q' during visualization to skip to next video\n")
    
    results = validator.validate_dataset(
        video_folder, 
        num_samples=10,  # Check 10 random videos
        visualize_first=True  # Show first video with keypoints
    )
    
    print("\nValidation complete!")


if __name__ == "__main__":
    main()
