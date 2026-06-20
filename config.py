"""VIPER configuration — all tunable parameters live here (SPEC section 7).

No hard-coded numbers anywhere else in src/. Changing VLM_BACKEND,
DEBATE_ENABLED, or any model id must require no other code changes.
"""

# ---- frame ingestion ----
FRAME_STRIDE = 15          # sample 1 of every N video frames
MAX_FRAMES = 20            # cap on extracted frames

# ---- location grounding ----
GROUNDING_MODE = "pretagged"        # "pretagged" (offline) | "vlm" (real models)
PRETAG_FILE = "data/zones.json"     # per-clip labeled A/B/obstacle regions

# ---- traversal animation ----
TRAVERSAL_FRAMES = 12      # frames the agent takes to move A->B
TRAVERSAL_OVERLAY = True   # render path/agent as overlay on real video frames

# ---- PIVOT ----
NUM_CANDIDATES = 5         # trajectories proposed per frame
MAX_TRAJECTORY_LEN = 10    # waypoints per trajectory
SELECT_TOP_K = 3           # candidates kept after VLM/fallback selection
SEED = 42                  # reproducible sampling

# ---- VLM (pipeline) ----
USE_VLM = False            # False = offline rule-based fallback
VLM_BACKEND = "mock"       # "mock" | "anthropic" | "molmo"
CLAUDE_MODEL = "claude-sonnet-4-6"   # cheap default for pipeline reasoning

# ---- VLMPC cost ----
COLLISION_PENALTY = 100.0
PATH_LENGTH_WEIGHT = 1.0
GOAL_DISTANCE_WEIGHT = 1.0

# ---- DEBATE ----
DEBATE_ENABLED = False     # off by default; turn on for the ensemble experiment
DEBATE_MAX_ROUNDS = 3
DEBATE_MODEL_A = "claude"  # first in the relay
DEBATE_MODEL_B = "molmo"   # second in the relay
DEBATE_CLAUDE_MODEL = "claude-sonnet-4-6"
MOLMO_MODEL_ID = "allenai/Molmo-7B-D-0924"
MOLMO_REVISION = "main"            # pin a commit hash for true reproducibility
MOLMO_TRUST_REMOTE_CODE = True     # Molmo ships custom modeling code
MOLMO_QUANTIZATION = "4bit"        # CUDA only
PARSE_RETRY = 1

# ---- evaluation ----
SUCCESS_DISTANCE_THRESHOLD = 0.10   # final pos within this of B (and no collision) = success

# ---- visualization ----
COLOR_CANDIDATE = (100, 160, 255)
COLOR_SELECTED = (86, 211, 100)
COLOR_REJECTED = (150, 150, 150)
COLOR_OBSTACLE = (239, 68, 68)
COLOR_GOAL = (250, 204, 21)
ARROW_THICKNESS = 2
GIF_FPS = 5
