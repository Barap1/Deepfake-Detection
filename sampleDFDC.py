import os
import shutil
import random
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

def collect_and_categorize_paths(dfdc_dir):
    print("Scanning and categorizing all video files from your local DFDC directory")
    real_videos_set = set()
    fake_videos_set = set()

    if not os.path.exists(dfdc_dir):
        print(f"doesnt exist: {dfdc_dir}")
        return [], []

    print(f"Scanning: {dfdc_dir}...")
    for i in range(50):
        part_path = os.path.join(dfdc_dir, f'dfdc_train_part_{i}')
        if os.path.exists(part_path):
            metadata_path = os.path.join(part_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                for filename, data in metadata.items():
                    video_path = os.path.join(part_path, filename)
                    if os.path.exists(video_path):
                        if data['label'] == 'REAL':
                            real_videos_set.add(os.path.abspath(video_path))
                        else:
                            fake_videos_set.add(os.path.abspath(video_path))
    
    all_real_videos = list(real_videos_set)
    all_fake_videos = list(fake_videos_set)

    print("scan summary")
    print(f"  - Found {len(all_real_videos)} total unique real videos.")
    print(f"  - Found {len(all_fake_videos)} total unique fake videos.")

    return all_real_videos, all_fake_videos

def compile_dataset(real_videos, fake_videos, output_path, num_fakes, num_reals):
    print(f"compiling final dataset of {num_reals} real and {num_fakes} fake videos ---")

    if not real_videos or not fake_videos:
        print("ERROR: No videos were found")
        return


    final_real_samples = random.sample(real_videos, num_reals)
    final_fake_samples = random.sample(fake_videos, num_fakes)

    real_output_dir = os.path.join(output_path, 'real')
    fake_output_dir = os.path.join(output_path, 'fake')
    os.makedirs(real_output_dir, exist_ok=True)
    os.makedirs(fake_output_dir, exist_ok=True)

    print(f"Copying {len(final_real_samples)} real videos...")
    for video_path in tqdm(final_real_samples, desc="Copying Real Videos"):
        try:
            shutil.copy(video_path, real_output_dir)
        except Exception as e:
            print(f"Could not copy {video_path}: {e}")

    print(f"Copying {len(final_fake_samples)} fake videos...")
    for video_path in tqdm(final_fake_samples, desc="Copying Fake Videos"):
        try:
            shutil.copy(video_path, fake_output_dir)
        except Exception as e:
            print(f"Could not copy {video_path}: {e}")

    print("\nDataset compilation complete!")
    print(f"Final dataset created at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="deepfake dataset")
    parser.add_argument('--dfdc_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./final_dataset_20k')
    parser.add_argument('--num_fakes', type=int, default=10000)
    parser.add_argument('--num_reals', type=int, default=10000)
    
    args = parser.parse_args()
    
    real_videos, fake_videos = collect_and_categorize_paths(args.dfdc_dir)
    compile_dataset(real_videos, fake_videos, args.output_dir, args.num_fakes, args.num_reals)

if __name__ == '__main__':
    main()
