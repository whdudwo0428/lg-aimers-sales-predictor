"""Model wrappers and training scripts.

Each subpackage under ``src.models`` corresponds to a specific
architecture (e.g. Fedformer, PatchTST, TimesFM, Autoformer).  The
subpackages expose a uniform API consisting of a ``model.py`` that
constructs the neural network, a ``config.py`` for hyperparameters, a
``train.py`` script encapsulating the training loop and a
``predict.py`` script for inference and submission generation.

These wrappers are intentionally minimal.  They import the
corresponding upstream implementation from the ``models/`` folder in
the project root and adapt it to the interface expected by the
pipeline.  If you wish to replace the baseline models with your own
implementation, simply modify the files within the relevant
subpackage.
"""

__all__ = [
    "model_fedformer.py",
    "model_patchtst.py",
    "model_timesfm.py",
    "model_autoformer.py",
]

