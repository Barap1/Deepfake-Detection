import os
import glob
import json
import argparse
import cv2
import dlib
from tqdm import tqdm

def get_videos_to_process(final_dataset_dir, celeb_df_dir):
    """
    Finds all videos in the compiled final_dataset and the Celeb-DF test set.
    """
    all_videos = []
    
    print(f"Scanning compiled training set: {final_dataset_dir}")
    for label in ['real', 'fake']:
        search_path = os.path.join(final_dataset_dir, label, '*.mp4')
        for video_file in glob.glob(search_path):
            all_videos.append({'path': video_file, 'split': 'train', 'label': label})
            
    print(f"Scanning Celeb-DF test set: {celeb_df_dir}")
    for label in ['real', 'fake']:
        folder = 'Celeb-real' if label == 'real' else 'Celeb-synthesis'
        search_path = os.path.join(celeb_df_dir, folder, '*.mp4')
        for video_file in glob.glob(search_path):
            all_videos.append({'path': video_file, 'split': 'test', 'label': label})
            
    return all_videos

def process_video_to_json(video_info, output_dir, detector):

    video_path = video_info['path']
    split = video_info['split']
    label = video_info['label']
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    final_output_dir = os.path.join(output_dir, "spatial", split, label)
    os.makedirs(final_output_dir, exist_ok=True)
    json_path = os.path.join(final_output_dir, f'{video_name}.json')
    
    if os.path.exists(json_path):
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video file {video_path}")
        return

    frame_boxes = []
    tracker = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if tracker is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 1)
            if not faces:
                frame_boxes.append(None) 
                continue
            face_rect = faces[0]
            tracker = dlib.correlation_tracker()
            tracker.start_track(frame, face_rect)
        else:
            tracker.update(frame)
            pos = tracker.get_position()
            face_rect = dlib.rectangle(int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
        frame_boxes.append([face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()])

    cap.release()
    
    with open(json_path, 'w') as f:
        json.dump(frame_boxes, f)

def main():
    parser = argparse.ArgumentParser(description="Preprocess a compiled deepfake dataset by detecting face bounding boxes.")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help="Path to the 'final_dataset_20k' directory created by the compilation script.")
    parser.add_argument('--celebs_dir', type=str, required=True,
                        help="Path to the Celeb-DF-v2 test set directory.")
    parser.add_argument('--output_dir', type=str, default='./processed_data',
                        help="Directory to save the output JSON files.")
    
    args = parser.parse_args()
    
    print("Initializing Dlib face detector")
    detector = dlib.get_frontal_face_detector()
    
    videos_to_process = get_videos_to_process(args.dataset_dir, args.celebs_dir)
    
    if not videos_to_process:
        print("ERROR: No videos found")
        return
        
    print(f"Found {len(videos_to_process)} total videos to preprocess...")
    
    for video_info in tqdm(videos_to_process, desc="Preprocessing Videos"):
        try:
            process_video_to_json(video_info, args.output_dir, detector)
        except Exception as e:
            print(f"error processing {video_info['path']}: {e}")

    print("\nPreprocessing complete!")
    print(f"Bounding box data saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
