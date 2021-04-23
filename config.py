from easydict import EasyDict as edict

__C = edict()

cfg = __C
# Other
__C.PROJECT_NAME = "WGAN-GP-Anime-Faces"
__C.PROJECT_VERSION_NAME = "Default-WGAN-GP"
# Global
__C.NUM_EPOCHS = 10
__C.LEARNING_RATE = 1e-4
__C.BATCH_SIZE = 64
__C.DATASET_SIZE = None
__C.CRITIC_ITERATIONS = 5
__C.LAMBDA_GP = 10
# CHANNELS
__C.IMG_SIZE = 64
__C.CHANNELS_IMG = 3
__C.Z_DIMENSION = 128
# Models
__C.FEATURES_DISC = 64
__C.FEATURES_GEN = 64
# Paths and saves
__C.SAVE_EACH_EPOCH = 5
__C.OUT_DIR = ''
__C.SAVE_CHECKPOINT_PATH = ''

# Display results
__C.NUM_SAMPLES = 64    # size grid for display images
__C.FREQ = 20
