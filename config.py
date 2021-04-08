from easydict import EasyDict as edict

__C = edict()

cfg = __C


__C.LEARNING_RATE = 1e-4
__C.BATCH_SIZE = 64
__C.IMAGE_SIZE = 64
__C.CHANNELS_IMG = 1
__C.Z_DIM = 100
__C.NUM_EPOCHS = 1
__C.FEATURES_CRITIC = 16
__C.FEATURES_GEN = 16
__C.CRITIC_ITERATIONS = 5
__C.LAMBDA_GRADIENT_PENALTY = 10
__C.OUT_DIR = ""
