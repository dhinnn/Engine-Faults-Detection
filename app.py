import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import joblib
import pickle
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model

custom_css = """
/* minimalistic css */
"""

class Config:
    """Configuration loaded from trained model parameters"""
    def __init__(self, config_path='model_config.pkl'):
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            
            self.CLASSES = config['CLASSES']
            self.NUM_CLASSES = config['NUM_CLASSES']
            self.SAMPLE_RATE = config['SAMPLE_RATE']
            self.DURATION = config['DURATION']
            self.N_MELS = config['N_MELS']
            self.N_FFT = config['N_FFT']
            self.HOP_LENGTH = config['HOP_LENGTH']
        else:
            self.CLASSES = ["Misfire", "Normal", "Rodknock", "ExhaustLeak"]
            self.NUM_CLASSES = 4
            self.SAMPLE_RATE = 22050
            self.DURATION = 5
            self.N_MELS = 128
            self.N_FFT = 2048
            self.HOP_LENGTH = 512

config = Config()

class ModelManager:
    """Manages loading and using pre-trained models"""
    
    def __init__(self):
        self.cnnlstm_model = None
        self.rf_model = None
        self.load_models()
    
    def load_models(self):
        try:
            if os.path.exists('best_cnnlstm_model.keras'):
                self.cnnlstm_model = load_model('best_cnnlstm_model.keras', compile=False)
        except Exception as e:
            print(f"Error loading CNN-LSTM model: {e}")
            self.cnnlstm_model = None

        try:
            if os.path.exists('random_forest_model.joblib'):
                self.rf_model = joblib.load('random_forest_model.joblib')
        except Exception as e:
            print(f"Error loading Random Forest model: {e}")
            self.rf_model = None
    
    def audio_to_spectrogram(self, audio_path):
        """Convert audio file to mel spectrogram"""
        try:
            signal, sr = librosa.load(
                audio_path, 
                sr=config.SAMPLE_RATE, 
                duration=config.DURATION, 
                res_type='kaiser_fast'
            )

            max_len = config.SAMPLE_RATE * config.DURATION
            if len(signal) < max_len:
                signal = np.pad(signal, (0, max_len - len(signal)), 'constant')
            else:
                signal = signal[:max_len]

            mel_spectrogram = librosa.feature.melspectrogram(
                y=signal,
                sr=sr,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                n_mels=config.N_MELS
            )

            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            return log_mel_spectrogram

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def predict_with_cnnlstm(self, audio_path):
        """Predict using CNN-LSTM model"""
        if self.cnnlstm_model is None:
            return "CNN-LSTM model not available", 0.0
        
        spectrogram = self.audio_to_spectrogram(audio_path)
        
        if spectrogram is None:
            return "Error processing audio file", 0.0
        
        sample_for_pred = np.expand_dims(spectrogram[..., np.newaxis], axis=0)
        
        prediction = self.cnnlstm_model.predict(sample_for_pred, verbose=0)
        predicted_class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        predicted_label = config.CLASSES[predicted_class_idx]
        return predicted_label, float(confidence)
    
    def predict_with_rf(self, audio_path):
        """Predict using Random Forest model"""
        if self.rf_model is None:
            return "Random Forest model not available", 0.0
        
        spectrogram = self.audio_to_spectrogram(audio_path)

        if spectrogram is None:
            return "Error processing audio file", 0.0

        reshaped_spectrogram = spectrogram.reshape(1, -1)

        prediction_proba = self.rf_model.predict_proba(reshaped_spectrogram)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        confidence = prediction_proba[predicted_class_idx]

        predicted_label = config.CLASSES[predicted_class_idx]
        return predicted_label, float(confidence)
    
    def predict_both_models(self, audio_path):
        """Get predictions from both models"""
        cnn_result = self.predict_with_cnnlstm(audio_path)
        rf_result = self.predict_with_rf(audio_path)
        
        result_text = f"""
        **CNN-LSTM Model:**
        Prediction: {cnn_result[0]}
        Confidence: {cnn_result[1]:.3f}
        
        **Random Forest Model:**
        Prediction: {rf_result[0]}
        Confidence: {rf_result[1]:.3f}
        """
        
        return result_text
    
    def visualize_spectrogram(self, audio_path):
        """Generate spectrogram visualization"""
        spectrogram = self.audio_to_spectrogram(audio_path)
        
        if spectrogram is None:
            return None
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            spectrogram, 
            sr=config.SAMPLE_RATE, 
            x_axis='time', 
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        
        plot_path = 'temp_spectrogram.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path

model_manager = ModelManager()

def predict_audio(audio_file):
    """Main prediction function for Gradio interface"""
    if audio_file is None:
        return "Please upload an audio file.", None
    
    try:
        results = model_manager.predict_both_models(audio_file)
        
        spectrogram_plot = model_manager.visualize_spectrogram(audio_file)
        
        return results, spectrogram_plot
        
    except Exception as e:
        return f"Error processing audio: {str(e)}", None

def predict_single_model(audio_file, model_type):
    """Predict using single model"""
    if audio_file is None:
        return "Please upload an audio file."
    
    try:
        if model_type == "CNN-LSTM":
            result = model_manager.predict_with_cnnlstm(audio_file)
        else:
            result = model_manager.predict_with_rf(audio_file)
        
        return f"Prediction: {result[0]}\nConfidence: {result[1]:.3f}"
        
    except Exception as e:
        return f"Error: {str(e)}"

def create_interface():
    with gr.Blocks(title="Engine Fault Detection", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown("# 🚗 Car Engine Fault Detection")
        gr.Markdown("Upload an audio file of a car engine to detect potential faults.")
        
        with gr.Tab("Both Models"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label="Upload Engine Audio", 
                        type="filepath"
                    )
                    predict_btn = gr.Button("Analyze Audio", variant="primary")
                
                with gr.Column():
                    results_output = gr.Markdown(label="Predictions")
                    spectrogram_output = gr.Image(label="Mel Spectrogram")
        
        with gr.Tab("Individual Models"):
            with gr.Row():
                with gr.Column():
                    audio_input2 = gr.Audio(
                        label="Upload Engine Audio", 
                        type="filepath"
                    )
                    model_choice = gr.Radio(
                        choices=["CNN-LSTM", "Random Forest"],
                        value="CNN-LSTM",
                        label="Select Model"
                    )
                    predict_btn2 = gr.Button("Predict", variant="primary")
                
                with gr.Column():
                    single_result = gr.Textbox(
                        label="Prediction Result", 
                        lines=3
                    )
        
        with gr.Tab("Model Info"):
            gr.Markdown(f"""
            ## Model Information
            
            **Fault Classes:** {', '.join(config.CLASSES)}
            
            **Audio Parameters:**
            - Sample Rate: {config.SAMPLE_RATE} Hz
            - Duration: {config.DURATION} seconds
            - Mel Bands: {config.N_MELS}
            
            **Models Available:**
            - CNN-LSTM: {'✅' if model_manager.cnnlstm_model else '❌'}
            - Random Forest: {'✅' if model_manager.rf_model else '❌'}
            """)
        
        predict_btn.click(
            predict_audio,
            inputs=[audio_input],
            outputs=[results_output, spectrogram_output]
        )
        
        predict_btn2.click(
            predict_single_model,
            inputs=[audio_input2, model_choice],
            outputs=[single_result]
        )
    
    return demo

if __name__ == "__main__":
    required_files = [
        'best_cnnlstm_model.keras',
        'random_forest_model.joblib',
        'model_config.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("⚠️  Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run the Colab export script and download the model files.")
        print("Place them in the same directory as this script.")
    
    demo = create_interface()
    
    print("🚀 Starting Engine Fault Detection App...")
    print("💡 The app will open in your browser automatically.")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=None,
        show_api=False,
        share=False,
        debug=False
    )