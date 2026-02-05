"""
Stuttering Detection Web Application
Deployed for academic demonstration
"""

import streamlit as st
import torch
import numpy as np
import librosa
import tempfile
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Stuttering Detection System",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎙️ Stuttering Detection System</h1>', unsafe_allow_html=True)
st.markdown("### An Interpretable, Weakly-Supervised System for Detecting Stuttering Events")

# Privacy notice
st.markdown("""
<div class="warning-box">
    <strong>🔒 Privacy Notice:</strong> Your audio file is processed locally and 
    <strong>automatically deleted</strong> after analysis. No data is stored.
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
**Research Project**  
BSc (Hons) Computer Science  
University of Westminster  

**Author:** W.W. Kavindu Mihiranga  
**Supervisor:** Mr. Rathesan Sivagnanalingam  

**Key Features:**
- 🎯 Real-time stuttering detection
- 📊 Interpretable AI explanations
- 🔍 Millisecond-level event localization
- 🔒 Privacy-preserving (auto-delete)
""")

# Model loading (cached)
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Import your model class
        import sys
        sys.path.insert(0, 'src')
        from models.neurosymbolic import NeuroSymbolicStutterDetectorCPU
        
        # Load config
        class Config:
            DEVICE = 'cpu'
            HIDDEN_DIM = 768
            NUM_CLASSES = 3  # Adjust based on your classes
            SAMPLE_RATE = 16000
            MAX_AUDIO_LENGTH = 48000
        
        config = Config()
        
        # Load model
        model = NeuroSymbolicStutterDetectorCPU(config, freeze_encoder=True)
        checkpoint = torch.load('models/checkpoints/cpu_final_model.pth', 
                              map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load label mappings
        with open('data/processed/label_mappings.json') as f:
            mappings = json.load(f)
        
        return model, config, mappings
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Preprocessing function
def preprocess_audio(audio_path, target_sr=16000, target_length=48000):
    """Preprocess audio for model input"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=target_sr)
        
        # Pad or trim to target length
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Normalize
        if audio.max() > 0:
            audio = audio / audio.max()
        
        return audio
    
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None

# Prediction function
def predict_stuttering(audio, model, config, mappings):
    """Run inference and return results"""
    try:
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            output = model(audio_tensor)
            
            # Get frame-level predictions
            frame_probs = torch.sigmoid(output).squeeze(0).cpu().numpy()
            
            # Get clip-level predictions (max pooling)
            clip_probs = frame_probs.max(axis=0)
        
        # Extract events
        events = extract_events(frame_probs, threshold=0.5)
        
        return {
            'frame_probs': frame_probs,
            'clip_probs': clip_probs,
            'events': events
        }
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def extract_events(frame_probs, threshold=0.5):
    """Extract discrete events from frame predictions"""
    events = []
    num_frames, num_classes = frame_probs.shape
    
    for class_idx in range(num_classes):
        probs = frame_probs[:, class_idx]
        binary = (probs > threshold).astype(int)
        
        if binary.sum() == 0:
            continue
        
        # Find connected components
        diff = np.diff(np.concatenate([[0], binary, [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            duration_ms = (end - start) * (3000 / num_frames)
            onset_ms = start * (3000 / num_frames)
            
            events.append({
                'class_idx': int(class_idx),
                'onset_ms': float(onset_ms),
                'offset_ms': float(onset_ms + duration_ms),
                'duration_ms': float(duration_ms),
                'confidence': float(probs[start:end].mean())
            })
    
    return events

def visualize_results(audio, frame_probs, events, mappings):
    """Create visualization of results"""
    
    # Get class names
    idx2label = mappings['idx2label']
    class_names = [idx2label[str(i)] for i in range(len(idx2label))]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Waveform with event markers
    ax = axes[0]
    time_audio = np.linspace(0, 3, len(audio))
    ax.plot(time_audio, audio, linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Audio Waveform with Detected Events', fontweight='bold')
    
    # Mark events
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    for event in events:
        onset_s = event['onset_ms'] / 1000
        offset_s = event['offset_ms'] / 1000
        class_idx = event['class_idx']
        ax.axvspan(onset_s, offset_s, alpha=0.3, 
                  color=colors[class_idx],
                  label=class_names[class_idx] if onset_s == events[0]['onset_ms'] else "")
    
    if events:
        ax.legend(loc='upper right')
    
    # 2. Frame-level predictions heatmap
    ax = axes[1]
    time_frames = np.linspace(0, 3, frame_probs.shape[0])
    
    im = ax.imshow(frame_probs.T, aspect='auto', origin='lower',
                   extent=[0, 3, 0, len(class_names)],
                   cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Event Type')
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_title('Frame-Level Predictions (Heatmap)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Probability')
    
    # 3. Detected events timeline
    ax = axes[2]
    if events:
        for i, event in enumerate(events):
            class_idx = event['class_idx']
            onset_s = event['onset_ms'] / 1000
            duration_s = event['duration_ms'] / 1000
            
            ax.barh(class_idx, duration_s, left=onset_s,
                   height=0.8, color=colors[class_idx],
                   alpha=0.7, edgecolor='black')
            
            # Add confidence label
            ax.text(onset_s + duration_s/2, class_idx,
                   f'{event["confidence"]:.2f}',
                   ha='center', va='center', fontsize=9,
                   color='black', fontweight='bold')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Event Type')
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlim([0, 3])
    ax.set_ylim([-0.5, len(class_names) - 0.5])
    ax.set_title('Detected Events (Timeline)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main app
def main():
    # Load model
    with st.spinner("Loading model... (this may take a moment)"):
        model, config, mappings = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check model files.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Instructions
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        1. **Record or upload** a 3-second audio clip of speech
        2. Click **"Analyze Speech"** to detect stuttering events
        3. View the **results and visualizations** below
        4. Your audio is **automatically deleted** after analysis
        
        **Supported formats:** WAV, MP3, M4A  
        **Optimal duration:** 3 seconds  
        **Sample rate:** 16 kHz (will be resampled automatically)
        """)
    
    # File upload
    st.markdown("### 🎤 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV, MP3, M4A)",
        type=['wav', 'mp3', 'm4a'],
        help="Upload a 3-second audio clip for analysis"
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Analyze button
        if st.button("🔍 Analyze Speech", type="primary"):
            
            with st.spinner("Processing audio... This may take 30-60 seconds."):
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                try:
                    # Preprocess
                    audio = preprocess_audio(tmp_path, config.SAMPLE_RATE, 
                                           config.MAX_AUDIO_LENGTH)
                    
                    if audio is None:
                        st.error("Failed to preprocess audio")
                        return
                    
                    # Predict
                    results = predict_stuttering(audio, model, config, mappings)
                    
                    if results is None:
                        st.error("Prediction failed")
                        return
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## 📊 Analysis Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    events = results['events']
                    num_events = len(events)
                    
                    if num_events > 0:
                        avg_duration = np.mean([e['duration_ms'] for e in events])
                        max_confidence = max([e['confidence'] for e in events])
                        event_types = len(set([e['class_idx'] for e in events]))
                    else:
                        avg_duration = 0
                        max_confidence = 0
                        event_types = 0
                    
                    col1.metric("Events Detected", num_events)
                    col2.metric("Avg Duration", f"{avg_duration:.0f} ms")
                    col3.metric("Max Confidence", f"{max_confidence:.2%}")
                    col4.metric("Event Types", event_types)
                    
                    # Event details
                    if num_events > 0:
                        st.markdown("### 🎯 Detected Events")
                        
                        idx2label = mappings['idx2label']
                        
                        for i, event in enumerate(events, 1):
                            class_name = idx2label[str(event['class_idx'])]
                            
                            with st.expander(f"Event {i}: {class_name} "
                                           f"({event['duration_ms']:.0f} ms)", 
                                           expanded=(i==1)):
                                col_a, col_b, col_c = st.columns(3)
                                col_a.write(f"**Type:** {class_name}")
                                col_b.write(f"**Onset:** {event['onset_ms']:.0f} ms")
                                col_c.write(f"**Confidence:** {event['confidence']:.2%}")
                    else:
                        st.info("✅ No stuttering events detected in this audio clip")
                    
                    # Visualization
                    st.markdown("### 📈 Visualization")
                    
                    fig = visualize_results(audio, results['frame_probs'], 
                                          events, mappings)
                    st.pyplot(fig)
                    
                    # Clinical interpretation
                    st.markdown("### 🏥 Clinical Interpretation")
                    
                    if num_events > 0:
                        events_per_min = (num_events / 3) * 60
                        
                        st.markdown(f"""
                        <div class="info-box">
                        <strong>Clinical Metrics:</strong>
                        <ul>
                            <li>Events per minute: <strong>{events_per_min:.1f}</strong></li>
                            <li>Average event duration: <strong>{avg_duration:.0f} ms</strong></li>
                            <li>Total event time: <strong>{sum(e['duration_ms'] for e in events):.0f} ms</strong></li>
                        </ul>
                        
                        <strong>Note:</strong> This is an automated analysis tool for research purposes. 
                        Clinical decisions should be made by qualified speech-language pathologists.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No clinical metrics to report (no events detected)")
                    
                    st.success("✅ Analysis complete!")
                
                finally:
                    # CRITICAL: Delete temporary file
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        st.info("🗑️ Audio file deleted for privacy")

if __name__ == "__main__":
    main()