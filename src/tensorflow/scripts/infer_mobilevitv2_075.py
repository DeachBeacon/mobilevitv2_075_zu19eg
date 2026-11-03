"""Run the adapted MobileViTv2-0.75 10-class TensorFlow 1.15 model on CPU."""
import argparse
import os
from _paths import PROJECT, TF_ROOT, TF_RECORDS, TF_WEIGHTS, INFERENCE, DEPLOY, REFERENCE_PB

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras_cv_attention_models import mobilevit


HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = str(TF_WEIGHTS)
REFERENCE = str(TF_RECORDS / "pytorch_reference.npz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Optional raw RGB image. It must already be 256x256.")
    args = parser.parse_args()

    model = mobilevit.MobileViT_V2_075_TF115(WEIGHTS)
    if args.image:
        from PIL import Image

        with Image.open(args.image) as image:
            rgb_uint8 = np.asarray(image.convert("RGB"))
        inputs = model.preprocess_input(rgb_uint8)
    else:
        # Demonstration input is already normalized float32; pass it through unchanged.
        with np.load(REFERENCE) as reference:
            inputs = reference["inputs_nhwc"][2:3].astype(np.float32)

    logits = model.predict(inputs, batch_size=1, verbose=0)
    print("logits shape: {}".format(tuple(logits.shape)))
    print("class index: {}".format(int(np.argmax(logits[0]))))
    print("logits: {}".format(logits[0].tolist()))


if __name__ == "__main__":
    main()
