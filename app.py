# run_app.py

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

warnings.filterwarnings("ignore")

# --- Configuration Class ---
class Config:
    """Holds model and preprocessing parameters."""
    SEQ_LENGTH = 20
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition (DeepfakeDetector) ---
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

# --- Preprocessing Pipeline ---
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
    """
    Reads a video and extracts face sequences for the model.
    Now supports different modes for speed vs. accuracy.
    """
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

# --- Main Application Logic ---
def main(args):
    print(f"Using device: {Config.DEVICE}")
    
    # Load face detectors
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

    def predict(video_path, mode):
        if video_path is None: return {"No video uploaded": 1.0}
        
        print(f"\nProcessing in '{mode}' mode...")

        # --- Set parameters based on selected mode ---
        if mode == "Super Fast":
            params = {'seq_length': 8, 'img_size': 160, 'num_chunks': 1, 'use_fast_detector': True, 'tta_flip': False}
        elif mode == "Deep Think":
            params = {'seq_length': 20, 'img_size': 224, 'num_chunks': 3, 'use_fast_detector': False, 'tta_flip': True}
        else: # Normal mode
            params = {'seq_length': 20, 'img_size': 224, 'num_chunks': 1, 'use_fast_detector': False, 'tta_flip': False}
        
        tensor_chunks = preprocess_video(video_path, face_detector, fast_face_detector, **params)
        
        if not tensor_chunks:
            return {"Error: Could not detect a face": 1.0}

        # --- Get prediction(s) ---
        predictions = []
        with torch.no_grad():
            for chunk in tensor_chunks:
                rgb_tensor = chunk.unsqueeze(0).to(Config.DEVICE)
                logit = model(rgb_tensor)
                predictions.append(torch.sigmoid(logit).item())
        
        # Average the predictions if multiple chunks were processed
        final_prob = np.mean(predictions)
        
        # Original labels: 0 for REAL, 1 for FAKE
        fake_prob = final_prob
        real_prob = 1 - final_prob
        
        print(f"Prediction complete. REAL: {real_prob:.2%}, FAKE: {fake_prob:.2%}")
        return {"FAKE": fake_prob, "REAL": real_prob}

    interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Video(label="Upload a Video"),
            gr.Radio(["Normal", "Super Fast", "Deep Think"], label="Processing Mode", value="Normal")
        ],
        outputs=gr.Label(num_top_classes=2, label="Prediction"),
        title="Deepfake Detector",
        description="Upload a video and choose a processing mode. 'Super Fast' is quick but less accurate. 'Deep Think' is slow but more accurate.",
        allow_flagging="never"
    )

    print("\nLaunching Gradio interface... Open the local URL in your browser.")
    interface.launch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a Gradio app for deepfake detection.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model file.")
    args = parser.parse_args()
    main(args)
