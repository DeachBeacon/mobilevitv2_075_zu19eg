"""Evaluate MobileViTv2-0.75 on a class-folder image dataset with TensorFlow 1.15."""
import argparse
import os
from _paths import PROJECT, TF_ROOT, TF_RECORDS, TF_WEIGHTS, INFERENCE, DEPLOY, REFERENCE_PB
import sys

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras_cv_attention_models import backend, mobilevit


DEFAULT_TEST_DIR = os.environ.get('MOBILEVIT_TEST_DIR', str(PROJECT/'data/calibration/jpegimages'))
DEFAULT_TRAIN_DIR = os.environ.get('MOBILEVIT_TRAIN_DIR', r"C:\transformerFPGA\MobileViT\Dataset\Driver\train_p")
DEFAULT_WEIGHTS = str(TF_WEIGHTS)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def class_directories(root):
    if not os.path.isdir(root):
        raise ValueError("Dataset directory does not exist: {}".format(root))
    return sorted([name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))])


def collect_samples(dataset_dir, class_names, limit=0):
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    samples = []
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        for current_dir, _, filenames in os.walk(class_dir):
            for filename in sorted(filenames):
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    samples.append((os.path.join(current_dir, filename), class_to_index[class_name]))
    samples.sort(key=lambda sample: sample[0].lower())
    if not samples:
        raise ValueError("No supported images found under {}".format(dataset_dir))
    return samples[:limit] if limit > 0 else samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR, help="Test root containing one subdirectory per class")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR, help="Training root used to verify the class-to-index mapping")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Strictly loaded TensorFlow H5 weights")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N sorted images; 0 means the full dataset")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("--batch-size must be positive and --limit cannot be negative")

    train_classes = class_directories(args.train_dir)
    test_classes = class_directories(args.test_dir)
    if train_classes != test_classes:
        raise ValueError(
            "Train/test class folders differ. train={}, test={}".format(train_classes, test_classes)
        )
    if len(train_classes) != 10:
        raise ValueError("Expected the 10 training classes, found {}: {}".format(len(train_classes), train_classes))

    # torchvision.datasets.ImageFolder uses sorted class directory names for class_to_idx.
    class_names = train_classes
    samples = collect_samples(args.test_dir, class_names, limit=args.limit)
    model = mobilevit.MobileViT_V2_075_TF115(args.weights)

    tf = backend.tf
    image_path = tf.compat.v1.placeholder(tf.string, shape=[], name="evaluation_image_path")
    image_contents = tf.io.read_file(image_path)
    decoded_rgb = tf.image.decode_image(image_contents, channels=3, name="decode_rgb")
    session = tf.compat.v1.keras.backend.get_session()

    class_counts = np.zeros((len(class_names),), dtype=np.int64)
    class_correct = np.zeros_like(class_counts)
    total_correct = 0
    evaluated = 0
    for start in range(0, len(samples), args.batch_size):
        batch_samples = samples[start : start + args.batch_size]
        batch_inputs = []
        batch_labels = []
        for path, label in batch_samples:
            rgb_uint8 = session.run(decoded_rgb, feed_dict={image_path: path})
            if rgb_uint8.dtype != np.uint8 or tuple(rgb_uint8.shape) != (256, 256, 3):
                raise ValueError(
                    "Expected 256x256 RGB image, got shape={} dtype={} at {}".format(rgb_uint8.shape, rgb_uint8.dtype, path)
                )
            batch_inputs.append(model.preprocess_input(rgb_uint8)[0])
            batch_labels.append(label)

        inputs = np.asarray(batch_inputs, dtype=np.float32)
        logits = model.predict(inputs, batch_size=len(batch_samples), verbose=0)
        if logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise RuntimeError("Inference returned non-finite logits or a non-float32 dtype")
        predictions = np.argmax(logits, axis=1)
        for label, prediction in zip(batch_labels, predictions):
            class_counts[label] += 1
            class_correct[label] += int(label == prediction)
            total_correct += int(label == prediction)
        evaluated += len(batch_samples)

    total_available = sum(
        1
        for class_name in class_names
        for current_dir, _, filenames in os.walk(os.path.join(args.test_dir, class_name))
        for filename in filenames
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    )
    print("Class mapping (ImageFolder order): {}".format({index: name for index, name in enumerate(class_names)}))
    print("Images: evaluated {} of {}".format(evaluated, total_available))
    print("Top-1 accuracy: {}/{} = {:.2f}%".format(total_correct, evaluated, 100.0 * total_correct / evaluated))
    for index, class_name in enumerate(class_names):
        count = int(class_counts[index])
        correct = int(class_correct[index])
        accuracy = 100.0 * correct / count if count else 0.0
        print("{}: {}/{} = {:.2f}%".format(class_name, correct, count, accuracy))
    if args.limit:
        print("NOTE: --limit was used; this is a partial-set accuracy, not the full test-set accuracy.")


if __name__ == "__main__":
    main()
