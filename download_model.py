"""
Download model checkpoint on first run
"""
import os
from huggingface_hub import hf_hub_download

def download_model():
    """Download model from Hugging Face Hub"""
    # You'll upload your model to HF Hub first
    # For now, we'll assume model is in models/checkpoints/
    
    model_path = "models/checkpoints/cpu_final_model.pth"
    
    if not os.path.exists(model_path):
        print("⚠️ Model not found. Please upload model to Hugging Face Hub.")
        print("For demo, using placeholder...")
        # In production, download from HF Hub:
        # model_path = hf_hub_download(repo_id="your-username/stuttering-model", 
        #                               filename="cpu_final_model.pth")
    
    return model_path

if __name__ == "__main__":
    download_model()