import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import cv2
import dlib
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gradio as gr
from tqdm import tqdm
import warnings
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

warnings.filterwarnings("ignore")

class Config:
    SEQ_LENGTH = 20
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True, rnn_hidden_size=128, rnn_num_layers=1, dropout_prob=0.5):
        super(DeepfakeDetector, self).__init__()
        if pretrained:
             weights = EfficientNet_B0_Weights.IMAGENET1K_V1
             self.cnn = efficientnet_b0(weights=weights)
        else:
             self.cnn = efficientnet_b0(weights=None)

        cnn_feature_dim = self.cnn.classifier[1].in_features
        self.cnn.classifier = nn.Identity()

        self.rnn = nn.GRU(
            input_size=cnn_feature_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x_reshaped = x.view(batch_size * seq_length, c, h, w)
        cnn_features = self.cnn(x_reshaped)
        rnn_input = cnn_features.view(batch_size, seq_length, -1)
        self.rnn.flatten_parameters()
        rnn_output, _ = self.rnn(rnn_input)
        last_time_step_output = rnn_output[:, -1, :]
        output = self.classifier(last_time_step_output)
        return output

def get_inference_transforms(image_size):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_inference_transforms_tta(image_size):
    # For "Deep Think" mode, adds a horizontal flip
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=1.0), # Always flip for TTA
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def preprocess_video(video_path, face_detector, fast_face_detector, seq_length, img_size, num_chunks=1, use_fast_detector=False, tta_flip=False):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None
        
    face_frames = []
    
    # Define chunk start points for "Deep Think" mode
    chunk_starts = np.linspace(0, total_frames - seq_length, num_chunks, dtype=int)
    
    all_chunks = []

    for i, start_frame in enumerate(chunk_starts):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_for_chunk = []
        frames_checked = 0
        
        # Collect frames for one chunk
        while len(frames_for_chunk) < seq_length and frames_checked < seq_length * 2:
            ret, frame = cap.read()
            if not ret: break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if use_fast_detector:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = fast_face_detector.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_cropped = frame_rgb[y:y+h, x:x+w]
                    frames_for_chunk.append(face_cropped)
            else:
                faces = face_detector(frame_rgb, 0)
                if len(faces) > 0:
                    d = faces[0]
                    face_cropped = frame_rgb[max(0, d.top()):d.bottom(), max(0, d.left()):d.right()]
                    frames_for_chunk.append(face_cropped)
            frames_checked += 1
            
        if not frames_for_chunk: continue

        # Pad if necessary
        while len(frames_for_chunk) < seq_length:
            frames_for_chunk.append(frames_for_chunk[-1])
        
        # Apply transforms
        transform = get_inference_transforms(img_size)
        transformed_chunk = [transform(image=f)['image'] for f in frames_for_chunk]
        all_chunks.append(torch.stack(transformed_chunk))
        
        # Apply TTA flip if enabled
        if tta_flip:
            transform_tta = get_inference_transforms_tta(img_size)
            tta_chunk = [transform_tta(image=f)['image'] for f in frames_for_chunk]
            all_chunks.append(torch.stack(tta_chunk))

    cap.release()
    return all_chunks if all_chunks else None

def visualize_detection_process(video_path, face_detector, fast_face_detector, seq_length, img_size, num_chunks=1, use_fast_detector=False, tta_flip=False):

    if video_path is None:
        return None, "No video uploaded"
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        return None, "Could not read video"
    
    # Step 1: Video Analysis
    analysis_text = f" **Video Analysis:**\n"
    analysis_text += f"- Total frames: {total_frames}\n"
    analysis_text += f"- FPS: {fps:.2f}\n"
    analysis_text += f"- Duration: {total_frames/fps:.2f} seconds\n"
    analysis_text += f"- Processing mode: {'Fast Detector' if use_fast_detector else 'Dlib Detector'}\n"
    analysis_text += f"- Sequence length: {seq_length} frames\n"
    analysis_text += f"- Number of chunks: {num_chunks}\n\n"
    
    # Step 2: Face Detection Visualization
    chunk_starts = np.linspace(0, total_frames - seq_length, num_chunks, dtype=int)
    
    # Create visualization images
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    face_detection_results = []
    processed_frames = []
    
    analysis_text += "🔍 **Face Detection Progress:**\n"
    
    for chunk_idx, start_frame in enumerate(chunk_starts):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_for_chunk = []
        frames_checked = 0
        faces_detected = 0
        
        # Collect frames for visualization (first 8 frames for display)
        viz_frames = []
        
        while len(frames_for_chunk) < seq_length and frames_checked < seq_length * 2:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Face detection
            face_found = False
            face_bbox = None
            
            if use_fast_detector:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = fast_face_detector.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_cropped = frame_rgb[y:y+h, x:x+w]
                    face_bbox = (x, y, w, h)
                    frames_for_chunk.append(face_cropped)
                    face_found = True
                    faces_detected += 1
            else:
                faces = face_detector(frame_rgb, 0)
                if len(faces) > 0:
                    d = faces[0]
                    face_cropped = frame_rgb[max(0, d.top()):d.bottom(), max(0, d.left()):d.right()]
                    face_bbox = (d.left(), d.top(), d.right() - d.left(), d.bottom() - d.top())
                    frames_for_chunk.append(face_cropped)
                    face_found = True
                    faces_detected += 1
            
            # Store for visualization (first 8 frames)
            if len(viz_frames) < 8:
                viz_frames.append({
                    'frame': frame_rgb,
                    'face_found': face_found,
                    'bbox': face_bbox,
                    'face_crop': face_cropped if face_found else None
                })
            
            frames_checked += 1
        
        analysis_text += f"  Chunk {chunk_idx + 1}: {faces_detected}/{frames_checked} faces detected\n"
        
        # Pad if necessary
        while len(frames_for_chunk) < seq_length:
            if frames_for_chunk:
                frames_for_chunk.append(frames_for_chunk[-1])
        
        processed_frames.extend(frames_for_chunk)
        
        # Visualize first 8 frames
        for i, viz_data in enumerate(viz_frames):
            if i < 8:
                ax = axes[i]
                ax.imshow(viz_data['frame'])
                ax.set_title(f"Frame {i+1}", fontsize=10)
                ax.axis('off')
                
                # Draw bounding box if face detected
                if viz_data['face_found'] and viz_data['bbox']:
                    x, y, w, h = viz_data['bbox']
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                           edgecolor='green', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x, y-5, 'Face Detected', color='green', fontsize=8, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                else:
                    ax.text(10, 30, 'No Face', color='red', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('Face Detection Results - First 8 Frames', fontsize=14, y=0.98)
    
    # Save the visualization
    face_detection_plot = plt.gcf()
    
    # Step 3: Feature Extraction Visualization
    if processed_frames:
        analysis_text += f"\n🧠 **Feature Extraction:**\n"
        analysis_text += f"- Total face crops extracted: {len(processed_frames)}\n"
        analysis_text += f"- Image size for processing: {img_size}x{img_size}\n"
        analysis_text += f"- CNN backbone: EfficientNet-B0\n"
        analysis_text += f"- RNN: GRU with 128 hidden units\n"
        
        # Show a sample of processed face crops
        fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
        axes2 = axes2.flatten()
        
        sample_indices = np.linspace(0, len(processed_frames)-1, 8, dtype=int)
        for i, idx in enumerate(sample_indices):
            if idx < len(processed_frames):
                # Apply transforms to show processed version
                transform = get_inference_transforms(img_size)
                processed_face = transform(image=processed_frames[idx])['image']
                
                # Convert tensor back to displayable format
                processed_face_np = processed_face.permute(1, 2, 0).numpy()
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                processed_face_np = processed_face_np * std + mean
                processed_face_np = np.clip(processed_face_np, 0, 1)
                
                axes2[i].imshow(processed_face_np)
                axes2[i].set_title(f"Processed Face {i+1}", fontsize=10)
                axes2[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('Processed Face Crops for CNN Feature Extraction', fontsize=14, y=0.98)
        processed_faces_plot = plt.gcf()
        
        analysis_text += f"- TTA (Test Time Augmentation): {'Enabled' if tta_flip else 'Disabled'}\n"
        
    else:
        analysis_text += "\n❌ **No faces detected in video!**\n"
        processed_faces_plot = None
    
    cap.release()
    
    return face_detection_plot, processed_faces_plot, analysis_text

def main(args):
    print(f"Using device: {Config.DEVICE}")

    face_detector = dlib.get_frontal_face_detector()
    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    fast_face_detector = cv2.CascadeClassifier(haar_cascade_path)
    print("Face detectors loaded.")

    try:
        model = DeepfakeDetector(pretrained=False)
        model.load_state_dict(torch.load(args.model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)
        model.eval()
        print(f"Model loaded successfully from {args.model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{args.model_path}'.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    def predict_with_visualization(video_path, mode):
        if video_path is None: 
            return {"No video uploaded": 1.0}, None, None, "No video uploaded"
        
        print(f"\nProcessing in '{mode}' mode with visualization...")

        if mode == "Super Fast":
            params = {'seq_length': 8, 'img_size': 160, 'num_chunks': 1, 'use_fast_detector': True, 'tta_flip': False}
        elif mode == "Deep Think":
            params = {'seq_length': 20, 'img_size': 224, 'num_chunks': 3, 'use_fast_detector': False, 'tta_flip': True}
        else: # Normal mode
            params = {'seq_length': 20, 'img_size': 224, 'num_chunks': 1, 'use_fast_detector': False, 'tta_flip': False}
        
        face_plot, processed_plot, analysis_text = visualize_detection_process(
            video_path, face_detector, fast_face_detector, **params
        )
        
        tensor_chunks = preprocess_video(video_path, face_detector, fast_face_detector, **params)
        
        if not tensor_chunks:
            return {"Error: Could not detect a face": 1.0}, face_plot, processed_plot, analysis_text + "\n❌ **Final Result: No faces detected - cannot make prediction**"

        predictions = []
        analysis_text += f"\n🔮 **Model Prediction:**\n"
        
        with torch.no_grad():
            for i, chunk in enumerate(tensor_chunks):
                rgb_tensor = chunk.unsqueeze(0).to(Config.DEVICE)
                logit = model(rgb_tensor)
                pred = torch.sigmoid(logit).item()
                predictions.append(pred)
                analysis_text += f"  Chunk {i+1} prediction: {pred:.4f}\n"
        
        final_prob = np.mean(predictions)
        
        fake_prob = final_prob
        real_prob = 1 - final_prob
        
        analysis_text += f"\n📊 **Final Results:**\n"
        analysis_text += f"- Average prediction: {final_prob:.4f}\n"
        analysis_text += f"- REAL probability: {real_prob:.2%}\n"
        analysis_text += f"- FAKE probability: {fake_prob:.2%}\n"
        analysis_text += f"- **Classification: {'FAKE' if fake_prob > 0.5 else 'REAL'}**"
        
        print(f"Prediction complete. REAL: {real_prob:.2%}, FAKE: {fake_prob:.2%}")
        return {"FAKE": fake_prob, "REAL": real_prob}, face_plot, processed_plot, analysis_text

    def predict(video_path, mode):
        if video_path is None: return {"No video uploaded": 1.0}
        
        print(f"\nProcessing in '{mode}' mode...")

        if mode == "Super Fast":
            params = {'seq_length': 8, 'img_size': 160, 'num_chunks': 1, 'use_fast_detector': True, 'tta_flip': False}
        elif mode == "Deep Think":
            params = {'seq_length': 20, 'img_size': 224, 'num_chunks': 3, 'use_fast_detector': False, 'tta_flip': True}
        else: # Normal mode
            params = {'seq_length': 20, 'img_size': 224, 'num_chunks': 1, 'use_fast_detector': False, 'tta_flip': False}
        
        tensor_chunks = preprocess_video(video_path, face_detector, fast_face_detector, **params)
        
        if not tensor_chunks:
            return {"Error: Could not detect a face": 1.0}

        predictions = []
        with torch.no_grad():
            for chunk in tensor_chunks:
                rgb_tensor = chunk.unsqueeze(0).to(Config.DEVICE)
                logit = model(rgb_tensor)
                predictions.append(torch.sigmoid(logit).item())
        
        final_prob = np.mean(predictions)
        
        fake_prob = final_prob
        real_prob = 1 - final_prob
        
        print(f"Prediction complete. REAL: {real_prob:.2%}, FAKE: {fake_prob:.2%}")
        return {"FAKE": fake_prob, "REAL": real_prob}

    # Create the main interface
    with gr.Blocks(title="Deepfake Detector", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔍 Deepfake Detection System")
        gr.Markdown("Upload a video to detect whether it contains deepfake content. Choose different processing modes based on your needs.")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload a Video")
                mode_input = gr.Radio(
                    ["Normal", "Super Fast", "Deep Think"], 
                    label="Processing Mode", 
                    value="Normal",
                    info="Super Fast: Quick but less accurate | Deep Think: Slow but more accurate"
                )
                
                with gr.Row():
                    predict_btn = gr.Button("🎯 Predict Only", variant="primary")
                    visualize_btn = gr.Button("🔍 Predict with Visualization", variant="secondary")
            
            with gr.Column():
                prediction_output = gr.Label(num_top_classes=2, label="Prediction Results")

                gr.Markdown("## 🎥 Example Videos")
                gr.Examples(
                    examples=[
                        ["Example1.mp4"],
                        ["Example2.mp4"],
                        ["Example3.mp4"],
                        ["Example4.mp4"]
                    ],
                    inputs=[video_input, mode_input],
                    label="Click on any example to load it",
                    examples_per_page=4
                )
        
        # Visualization outputs (hidden by default)
        with gr.Row(visible=False) as viz_row:
            with gr.Column():
                face_detection_plot = gr.Plot(label="Face Detection Results")
                analysis_text = gr.Textbox(
                    label="Detailed Analysis", 
                    lines=15, 
                    max_lines=20,
                    interactive=False
                )
            with gr.Column():
                processed_faces_plot = gr.Plot(label="Processed Face Crops")
        
        # Event handlers
        predict_btn.click(
            fn=predict,
            inputs=[video_input, mode_input],
            outputs=[prediction_output]
        )
        
        visualize_btn.click(
            fn=predict_with_visualization,
            inputs=[video_input, mode_input],
            outputs=[prediction_output, face_detection_plot, processed_faces_plot, analysis_text]
        ).then(
            fn=lambda: gr.Row(visible=True),
            outputs=[viz_row]
        )

    print("\nLaunching Gradio interface... Open the local URL in your browser.")
    interface.launch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a Gradio app for deepfake detection.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model file.")
    args = parser.parse_args()
    main(args)
