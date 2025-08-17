# Configuration for experiments

# Core columns in your CSV
TIMESTAMP_COL = "timestamp"          # pandas-parsable datetime string
SITE_COL      = "site_id"            # optional; set to None if not available
LABEL_COL     = "label"              # 0/1 larval presence

# List the meteorological features to use (ordered)
FEATURE_COLS = [
    # examples; replace with your exact columns
    "MORNING_TEMP", "EVENING_TEMP", "NIGHT_TEMP", "MIN_TEMP", "MAX_TEMP",
    "HUMIDITY", "PRESSURE_AFTERNOON", "PRECIPITATION",
    "MAX_WIND_SPEED", "MAX_WIND_DIRECTION", "CLOUD_COVER"
]

# Sequence/window parameters
L_WINDOW = 3     # default window length L
LEAD_H   = 0     # label lead-time (0 = nowcasting; set >0 if you have future labels)

# Split parameters (blocked by time)
TRAIN_PROP = 0.6
VAL_PROP   = 0.2  # test will be 1 - TRAIN_PROP - VAL_PROP

# SMOTE–Tomek toggle (applies to TRAIN ONLY, after split)
USE_SMOTE_TOMEK = True
SMOTE_RANDOM_STATE = 42

# Random seeds
GLOBAL_SEED = 42

# Training parameters
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-4
PATIENCE = 12

# Baseline LSTM/BiLSTM widths
LSTM_UNITS = 64

# MHA settings
NUM_HEADS = 6
KEY_DIM   = 64

# Convolution settings
CONV_FILTERS = 128
CONV_KERNEL  = 3

# Threshold strategy: "f1" or "youden"
THRESHOLD_STRATEGY = "f1"
NUM_ECE_BINS = 10

# Paths
OUTPUT_DIR = "outputs"
