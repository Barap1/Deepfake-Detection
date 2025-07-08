import os
import glob
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from scipy.fftpack import dct

def get_files_to_process(spatial_data_dir):
    """Finds all JSON files created by the previous spatial preprocessing script."""
    all_json_files = []
    print(f"Scanning for JSON files in: {spatial_data_dir}")
    for f in glob.glob(os.path.join(spatial_data_dir, '**', '*.json'), recursive=True):
        all_json_files.append(f)
    return all_json_files

def find_original_video(json_path, final_dataset_dir, celebs_dir):
    """Finds the original MP4 file corresponding to a JSON file."""
    video_name = os.path.splitext(os.path.basename(json_path))[0]
    split, label = os.path.normpath(json_path).split(os.sep)[-3:-1]
    
    if split == 'train':
        path = os.path.join(final_dataset_dir, label, f"{video_name}.mp4")
        if os.path.exists(path): return path
    elif split == 'test':
        folder = 'Celeb-real' if label == 'real' else 'Celeb-synthesis'
        path = os.path.join(celebs_dir, folder, f"{video_name}.mp4")
        if os.path.exists(path): return path
    return None

def extract_pde_features(json_path, video_path, output_dir):
    """
    Reads a video and its bounding box JSON to extract the three feature streams
    for the PDE-Net and saves them to a single compressed .npz file.
    """
    video_name = os.path.splitext(os.path.basename(json_path))[0]
    split, label = os.path.normpath(json_path).split(os.sep)[-3:-1]
    
    final_output_dir = os.path.join(output_dir, "pde_features", split, label)
    os.makedirs(final_output_dir, exist_ok=True)
    npz_path = os.path.join(final_output_dir, f'{video_name}.npz')

    if os.path.exists(npz_path): return
    
    try:
        with open(json_path, 'r') as f: bounding_boxes = json.load(f)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return

        rgb_stream, freq_stream, grad_stream = [], [], []
        
        valid_frames_info = [{'idx': i, 'box': b} for i, b in enumerate(bounding_boxes) if b]
        
        NUM_FRAMES = 32
        if len(valid_frames_info) < NUM_FRAMES:
            cap.release()
            return

        indices_to_process = np.linspace(0, len(valid_frames_info) - 1, NUM_FRAMES, dtype=int)
        
        for i in indices_to_process:
            info = valid_frames_info[i]
            frame_idx, box = info['idx'], info['box']
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            x1, y1, x2, y2 = box
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0: continue
            resized_face = cv2.resize(face_crop, (224, 224))
            
            # 1. RGB Stream
            rgb_stream.append(cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB))
            
            # 2. Frequency Stream
            gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            freq_data = dct(dct(gray_face.T, norm='ortho').T, norm='ortho')
            freq_stream.append(freq_data)
            
            # 3. Gradient Stream
            sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=5)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            grad_stream.append(gradient_magnitude)
            
        cap.release()
        
        if len(rgb_stream) == NUM_FRAMES:
            np.savez_compressed(npz_path, 
                                rgb=np.array(rgb_stream, dtype=np.uint8),
                                frequency=np.array(freq_stream, dtype=np.float16),
                                gradient=np.array(grad_stream, dtype=np.float16))
    except Exception as e:
        print(f"ERROR processing {video_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract PDE-Net features from pre-calculated bounding boxes.")
    parser.add_argument('--spatial_dir', type=str, default='./processed_data/spatial', help="Path to the folder with bounding box JSONs.")
    parser.add_argument('--dataset_dir', type=str, default='./final_dataset_20k', help="Path to the compiled video dataset.")
    parser.add_argument('--celebs_dir', type=str, default='./Celeb-DF-v2', help="Path to the Celeb-DF-v2 test set directory.")
    parser.add_argument('--output_dir', type=str, default='./processed_data', help="Root directory to save final .npz files.")
    args = parser.parse_args()

    json_files = get_files_to_process(args.spatial_dir)
    print(f"Found {len(json_files)} videos with bounding boxes to process for features...")

    for json_path in tqdm(json_files, desc="Extracting PDE-Net Features"):
        original_video_path = find_original_video(json_path, args.dataset_dir, args.celebs_dir)
        if original_video_path:
            extract_pde_features(json_path, original_video_path, args.output_dir)
        else:
            print(f"Warning: Could not find original video for {os.path.basename(json_path)}")

    print("\nAdvanced feature extraction complete!")

if __name__ == '__main__':
    main()
