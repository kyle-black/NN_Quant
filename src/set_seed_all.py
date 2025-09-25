# seeds.py (or inline at top of your scripts)
import os
import random
import numpy as np

def seed_everything(seed: int = 42, deterministic_tf: bool = True):
    # 1) Pure-Python & hashing
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # 2) NumPy
    np.random.seed(seed)

    # 3) TensorFlow / Keras (set env BEFORE importing TF when possible)
    if deterministic_tf:
        # Ensure deterministic GPU/CPU kernels where possible
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        # Disable OneDNN graph optimizations (they can introduce minor non-determinism on CPU)
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Now import TF after env vars are set
    import tensorflow as tf
    try:
        # TF 2.12+: also available via tf.keras.utils.set_random_seed
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)

    # (Optional but recommended) single-threaded math for strict determinism
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except Exception:
        pass

    # If you see cuDNN nondeterministic warnings, ensure you aren’t using
    # ops that lack deterministic kernels in your TF version.