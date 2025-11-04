"""
Advanced Face Recognition System
Author: BrillConsulting
Description: Deep learning-based face recognition with enrollment and identification
"""

import cv2
import numpy as np
import face_recognition
from pathlib import Path
import pickle
import argparse
from typing import List, Tuple, Dict
import time
from collections import defaultdict


class FaceRecognitionSystem:
    """
    Complete face recognition system with enrollment and real-time identification
    """

    def __init__(self, database_path: str = 'face_database.pkl',
                 tolerance: float = 0.6):
        """
        Initialize face recognition system

        Args:
            database_path: Path to save/load face encodings database
            tolerance: Face matching tolerance (lower = stricter)
        """
        self.database_path = Path(database_path)
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_database()

    def enroll_face(self, image_path: str, person_name: str) -> bool:
        """
        Enroll a new face into the system

        Args:
            image_path: Path to face image
            person_name: Name of the person

        Returns:
            True if enrollment successful
        """
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            print(f"❌ No face detected in {image_path}")
            return False

        if len(face_locations) > 1:
            print(f"⚠️  Multiple faces detected. Using first face.")

        # Get face encoding
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_encoding = face_encodings[0]

        # Add to database
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(person_name)

        print(f"✅ Enrolled {person_name} successfully")
        return True

    def enroll_from_directory(self, directory_path: str) -> int:
        """
        Enroll all faces from directory (one person per subdirectory)

        Args:
            directory_path: Path to directory with subdirectories for each person

        Returns:
            Number of faces enrolled
        """
        directory = Path(directory_path)
        count = 0

        for person_dir in directory.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            print(f"\n📁 Enrolling {person_name}...")

            for image_path in person_dir.glob('*'):
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    if self.enroll_face(str(image_path), person_name):
                        count += 1

        self.save_database()
        print(f"\n🎉 Enrolled {count} faces from {len(set(self.known_face_names))} people")
        return count

    def recognize_faces(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Recognize faces in an image

        Args:
            image: Input image (BGR format from OpenCV)

        Returns:
            Annotated image and list of recognized faces
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        recognized_faces = []
        annotated_image = image.copy()

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=self.tolerance
            )
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )

            name = "Unknown"
            confidence = 0.0

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]

            # Extract face location
            top, right, bottom, left = face_location

            recognized_faces.append({
                'name': name,
                'confidence': confidence,
                'location': (left, top, right, bottom)
            })

            # Draw rectangle
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(annotated_image, (left, top), (right, bottom), color, 2)

            # Draw label
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else name
            cv2.rectangle(annotated_image, (left, bottom - 35),
                         (right, bottom), color, cv2.FILLED)
            cv2.putText(annotated_image, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return annotated_image, recognized_faces

    def recognize_video(self, video_source: int = 0,
                       output_path: str = None) -> None:
        """
        Real-time face recognition from video/webcam

        Args:
            video_source: Camera ID or video path
            output_path: Optional path to save output
        """
        cap = cv2.VideoCapture(video_source)

        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Performance tracking
        frame_count = 0
        process_every_n_frames = 2  # Process every Nth frame for speed
        face_counts = defaultdict(int)

        print("🎥 Starting face recognition... Press 'q' to quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every Nth frame
            if frame_count % process_every_n_frames == 0:
                start_time = time.time()

                # Resize for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                annotated_frame, faces = self.recognize_faces(small_frame)

                # Scale back up
                annotated_frame = cv2.resize(annotated_frame,
                                            (frame.shape[1], frame.shape[0]))

                process_time = time.time() - start_time
                fps = 1 / process_time if process_time > 0 else 0

                # Track face counts
                for face in faces:
                    if face['name'] != "Unknown":
                        face_counts[face['name']] += 1

                # Add FPS counter
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Add face count
                cv2.putText(annotated_frame, f"Faces: {len(faces)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                annotated_frame = frame

            if output_path:
                out.write(annotated_frame)

            cv2.imshow('Face Recognition', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print statistics
        print("\n📊 Recognition Statistics:")
        for name, count in sorted(face_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {count} frames")

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

    def save_database(self) -> None:
        """Save face encodings database to disk"""
        with open(self.database_path, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
        print(f"💾 Database saved to {self.database_path}")

    def load_database(self) -> None:
        """Load face encodings database from disk"""
        if self.database_path.exists():
            with open(self.database_path, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f"✅ Loaded {len(self.known_face_names)} faces from database")
        else:
            print("ℹ️  No existing database found. Starting fresh.")

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        unique_people = len(set(self.known_face_names))
        return {
            'total_faces': len(self.known_face_names),
            'unique_people': unique_people,
            'people': list(set(self.known_face_names))
        }


def main():
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--mode', type=str, default='recognize',
                       choices=['enroll', 'enroll-dir', 'recognize', 'webcam', 'stats'],
                       help='Operation mode')
    parser.add_argument('--source', type=str, default='0',
                       help='Image/video path or camera ID')
    parser.add_argument('--name', type=str, help='Person name for enrollment')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--tolerance', type=float, default=0.6,
                       help='Face matching tolerance')
    parser.add_argument('--database', type=str, default='face_database.pkl',
                       help='Database path')

    args = parser.parse_args()

    # Initialize system
    system = FaceRecognitionSystem(database_path=args.database,
                                   tolerance=args.tolerance)

    if args.mode == 'enroll':
        if not args.name:
            print("❌ --name required for enrollment")
            return
        system.enroll_face(args.source, args.name)
        system.save_database()

    elif args.mode == 'enroll-dir':
        system.enroll_from_directory(args.source)

    elif args.mode == 'recognize':
        image = cv2.imread(args.source)
        if image is None:
            print(f"❌ Could not load image from {args.source}")
            return

        annotated_image, faces = system.recognize_faces(image)

        print(f"\n👤 Recognized {len(faces)} face(s):")
        for i, face in enumerate(faces, 1):
            print(f"  {i}. {face['name']} (confidence: {face['confidence']:.2f})")

        if args.output:
            cv2.imwrite(args.output, annotated_image)
            print(f"💾 Saved to {args.output}")

        cv2.imshow('Face Recognition', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == 'webcam':
        source = int(args.source) if args.source.isdigit() else args.source
        system.recognize_video(source, output_path=args.output)

    elif args.mode == 'stats':
        stats = system.get_statistics()
        print(f"\n📊 Database Statistics:")
        print(f"  Total faces: {stats['total_faces']}")
        print(f"  Unique people: {stats['unique_people']}")
        print(f"  People: {', '.join(stats['people'])}")


if __name__ == "__main__":
    main()
