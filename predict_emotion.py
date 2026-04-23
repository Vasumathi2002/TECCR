"""
TECCR Emotion Prediction Module
Loads trained model and predicts emotions from Tamil text
"""

import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TECCREmotionPredictor:
    """TECCR Emotion Prediction Engine"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.primary_encoder = None
        self.secondary_encoder = None
        self.metadata = None
        self.loaded = False
        
    def load_models(self):
        """Load all required model components"""
        try:
            print("🔄 Loading TECCR models...")
            
            # Load trained model
            model_path = os.path.join(self.model_dir, "teccr_model.h5")
            self.model = load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
            
            # Load tokenizer
            tokenizer_path = os.path.join(self.model_dir, "tokenizer.pkl")
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✅ Tokenizer loaded")
            
            # Load encoders
            primary_encoder_path = os.path.join(self.model_dir, "primary_encoder.pkl")
            with open(primary_encoder_path, 'rb') as f:
                self.primary_encoder = pickle.load(f)
            print(f"✅ Primary encoder loaded")
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, "model_metadata.pkl")
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Metadata loaded")
            
            self.loaded = True
            print("🎉 All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def preprocess_text(self, text):
        """Convert text to model input format"""
        if not self.loaded:
            raise Exception("Models not loaded. Call load_models() first.")
        
        # Convert text to sequence
        sequence = self.tokenizer.texts_to_sequences([text])
        
        # Pad sequence
        padded = pad_sequences(
            sequence,
            maxlen=self.metadata['max_len'],
            padding='post',
            truncating='post'
        )
        
        return padded
    
    def predict(self, text):
        """
        Predict emotion from Tamil text
        
        Args:
            text (str): Tamil comment text
            
        Returns:
            dict: Prediction results with emotions and confidence
        """
        if not self.loaded:
            self.load_models()
        
        try:
            # Preprocess input
            input_data = self.preprocess_text(text)
            
            # Make prediction
            prediction = self.model.predict(input_data, verbose=0)
            
            # Extract primary and secondary predictions (squeeze batch dimension)
            primary_pred = prediction[0][0]  # Shape: (num_classes,)
            secondary_pred = prediction[1][0]  # Shape: (num_classes,)
            
            # Get primary emotion
            primary_idx = np.argmax(primary_pred)
            primary_confidence = float(primary_pred[primary_idx])
            primary_emotion = self.primary_encoder.classes_[primary_idx]
            
            # Get top 3 emotions with probabilities
            top_3_idx = np.argsort(primary_pred)[-3:][::-1]
            top_emotions = [
                {
                    'emotion': self.primary_encoder.classes_[idx],
                    'confidence': float(primary_pred[idx])
                }
                for idx in top_3_idx
            ]
            
            # Generate TECCR context (simple rule-based for now)
            teccr_context = self._generate_context(primary_emotion, primary_confidence, text)
            
            result = {
                'text': text,
                'primary_emotion': primary_emotion,
                'confidence': primary_confidence,
                'secondary_emotion': top_emotions[1]['emotion'] if len(top_emotions) > 1 else None,
                'teccr_context': teccr_context,
                'all_emotions': top_emotions,
                'success': True
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': text
            }
    
    def _generate_context(self, emotion, confidence, text):
        """Generate TECCR cultural context explanation"""
        context_map = {
            'Sadness': 'Implicit despair detected - Cultural expression of suppressed emotions',
            'Anger': 'Frustration pattern - Indirect aggression common in Tamil discourse',
            'Frustration': 'Persistent dissatisfaction - Contextual disappointment markers',
            'Sarcasm': 'Ironic expression - Cultural humor masking true sentiment',
            'Joy': 'Positive sentiment - Genuine happiness expression',
            'Neutral': 'Balanced tone - No strong emotional markers',
            'Fear': 'Anxiety indicators - Uncertainty and apprehension',
            'Surprise': 'Unexpected reaction - Shock or amazement markers'
        }
        
        base_context = context_map.get(emotion, 'General emotional expression detected')
        
        # Add confidence level
        if confidence > 0.85:
            confidence_level = "High confidence"
        elif confidence > 0.65:
            confidence_level = "Moderate confidence"
        else:
            confidence_level = "Low confidence"
        
        return f"{base_context} ({confidence_level})"
    
    def predict_batch(self, texts):
        """Predict emotions for multiple texts"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def get_model_info(self):
        """Get model information and statistics"""
        if not self.loaded:
            self.load_models()
        
        return {
            'model_loaded': self.loaded,
            'primary_classes': self.metadata['primary_classes'],
            'num_classes': self.metadata['num_primary_classes'],
            'vocabulary_size': self.metadata['max_words'],
            'max_sequence_length': self.metadata['max_len'],
            'training_accuracy': self.metadata['final_primary_accuracy'],
            'validation_accuracy': self.metadata['final_primary_val_accuracy']
        }


# Global predictor instance (singleton pattern)
_predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = TECCREmotionPredictor()
        _predictor.load_models()
    return _predictor

def predict_emotion(text):
    """Convenience function for single prediction"""
    predictor = get_predictor()
    return predictor.predict(text)

# Test function
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TESTING TECCR EMOTION PREDICTOR")
    print("=" * 60)
    
    # Initialize predictor
    predictor = TECCREmotionPredictor()
    predictor.load_models()
    
    # Test samples (Tamil text)
    test_samples = [
        "இந்த உலகம் உண்மையில்ல நம்ம மனச் தான் பொய்யா?",
        "அப்படிப்பா என்ன வாழ்க்கை இது",
        "சூப்பர் படம், நன்றாக இருந்தது!",
        "கஷ்டம் தான், ஆனா என்ன செய்றது",
        "முடிவே இல்லை என்ற உணர்வு",
        "சிறு மாற்றம் கூட பெரிய நம்பிக்கைகளை தரும்",
        "என்னோட நண்பர்கள் எல்லாம் என்னை புரிஞ்சுக்க முடியலை"
    ]
    
    print("\n📝 Testing predictions:\n")
    for i, text in enumerate(test_samples, 1):
        print(f"Test {i}: {text}")
        result = predictor.predict(text)
        
        if result['success']:
            print(f"   Primary: {result['primary_emotion']} ({result['confidence']:.2%})")
            print(f"   Context: {result['teccr_context']}")
        else:
            print(f"   Error: {result['error']}")
        print()
    
    # Show model info
    info = predictor.get_model_info()
    print("\n" + "=" * 60)
    print("📊 MODEL INFORMATION")
    print("=" * 60)
    for key, value in info.items():
        print(f"{key}: {value}")