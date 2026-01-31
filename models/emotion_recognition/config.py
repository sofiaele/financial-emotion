"""Configuration for emotion recognition model"""

# Model architecture
BASE_MODEL = "trillsson3"  # Based on EfficientNetV2-S
EMBEDDING_DIM = 1024
NUM_EMOTION_CLASSES = 4  # angry, happy, neutral, sad
NUM_VALENCE_CLASSES = 3  # positive, neutral, negative
NUM_AROUSAL_CLASSES = 3  # weak, neutral, strong

# Audio processing
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FFT = 512

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
GRADIENT_CLIP = 1.0

# Dataset paths (to be set by user)
IEMOCAP_PATH = "/path/to/IEMOCAP"  # User needs to download
MSP_PODCAST_PATH = "/path/to/MSP-Podcast"  # User needs to download

# IEMOCAP splits
IEMOCAP_VAL_SESSION = "Session3"
IEMOCAP_TEST_SESSION = "Session5"

# Label mappings
EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]
VALENCE_LABELS = ["negative", "neutral", "positive"]
AROUSAL_LABELS = ["weak", "neutral", "strong"]

# Device
DEVICE = "cuda"  # Will fallback to "cpu" if CUDA not available

# Model save path
MODEL_CHECKPOINT_DIR = "../data/emotion_models"
