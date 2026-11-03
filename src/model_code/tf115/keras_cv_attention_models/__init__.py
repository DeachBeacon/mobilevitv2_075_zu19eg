"""Public package entry with a minimal import path for TensorFlow 1.15."""
from keras_cv_attention_models import backend
from keras_cv_attention_models.version import __version__

_is_tensorflow_1 = backend.is_tensorflow_backend and str(backend.tf.__version__).split(".")[0] == "1"
if _is_tensorflow_1:
    # Python 3.6 has no module-level __getattr__; import the supported model API directly.
    from keras_cv_attention_models import mobilevit
else:
    raise RuntimeError('This release bundles only the TensorFlow 1.15 MobileViT dependency closure. Use the tf115 PC environment.')
