"""Pinned inputs and model-level constants."""

MODEL_REPO_ID = "lerobot/smolvla_base"
MODEL_REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"
LOCAL_CHECKPOINT = "checkpoints/raw/smolvla/base"
# The released base checkpoint has no separate dataset pin. Stats in the file
# are the SO-100 buffers shipped with the pretrained policy.
DATASET_REPO_ID = ""
DATASET_REVISION = ""

ACTION_DIM = 6
PADDED_ACTION_DIM = 32
PADDED_STATE_DIM = 32
CHUNK_SIZE = 50
NUM_STEPS = 10
TOKENIZER_MAX_LENGTH = 48
RESIZE_HEIGHT = 512
RESIZE_WIDTH = 512

# smolvla_base lists three cameras and sets empty_cameras to 0, so a missing
# camera is omitted instead of being replaced by a masked placeholder.
IMAGE_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)
