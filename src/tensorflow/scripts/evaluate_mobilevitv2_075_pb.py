"""Evaluate the frozen MobileViTv2 GraphDef on a class-folder dataset in TensorFlow 1.15."""
import argparse
import os
from _paths import PROJECT, TF_ROOT, TF_RECORDS, TF_WEIGHTS, INFERENCE, DEPLOY, REFERENCE_PB

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


DEFAULT_TEST_DIR = os.environ.get('MOBILEVIT_TEST_DIR', str(PROJECT/'data/calibration/jpegimages'))
DEFAULT_TRAIN_DIR = os.environ.get('MOBILEVIT_TRAIN_DIR', r"C:\transformerFPGA\MobileViT\Dataset\Driver\train_p")
DEFAULT_PB = str(INFERENCE)
DEFAULT_INPUT_TENSOR = "input_1:0"
DEFAULT_OUTPUT_TENSOR = "predictions/BiasAdd:0"
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


def count_images(dataset_dir, class_names):
    return sum(
        1
        for class_name in class_names
        for current_dir, _, filenames in os.walk(os.path.join(dataset_dir, class_name))
        for filename in filenames
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    )


def preprocess_rgb_uint8(rgb_uint8):
    if rgb_uint8.dtype != np.uint8 or tuple(rgb_uint8.shape) != (256, 256, 3):
        raise ValueError("Expected 256x256 RGB uint8 image, got shape={} dtype={}".format(rgb_uint8.shape, rgb_uint8.dtype))
    image = rgb_uint8.astype(np.float32) / np.float32(255.0)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return ((image - mean) / std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR, help="Test root containing one subdirectory per class")
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR, help="Training root used to verify class-to-index mapping")
    parser.add_argument("--pb", default=DEFAULT_PB, help="Frozen TensorFlow GraphDef file")
    parser.add_argument("--input-tensor", default=DEFAULT_INPUT_TENSOR)
    parser.add_argument("--output-tensor", default=DEFAULT_OUTPUT_TENSOR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N sorted images; 0 means all images")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("--batch-size must be positive and --limit cannot be negative")
    if not os.path.isfile(args.pb):
        raise ValueError("Frozen PB file does not exist: {}".format(args.pb))

    train_classes = class_directories(args.train_dir)
    test_classes = class_directories(args.test_dir)
    if train_classes != test_classes:
        raise ValueError("Train/test class folders differ. train={}, test={}".format(train_classes, test_classes))
    if len(train_classes) != 10:
        raise ValueError("Expected the 10 training classes, found {}: {}".format(len(train_classes), train_classes))
    class_names = train_classes  # Same sorted mapping as torchvision.datasets.ImageFolder.
    samples = collect_samples(args.test_dir, class_names, limit=args.limit)

    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(args.pb, "rb") as pb_file:
        graph_def.ParseFromString(pb_file.read())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
        image_path = tf.compat.v1.placeholder(tf.string, shape=[], name="evaluation_image_path")
        image_contents = tf.io.read_file(image_path)
        decoded_rgb = tf.image.decode_image(image_contents, channels=3, name="decode_rgb")

    input_tensor = graph.get_tensor_by_name(args.input_tensor)
    output_tensor = graph.get_tensor_by_name(args.output_tensor)
    input_shape = input_tensor.shape.as_list()
    if (input_tensor.dtype != tf.float32 or len(input_shape) != 4
            or input_shape[0] not in (None, 1) or input_shape[1:] != [256, 256, 3]):
        raise ValueError("Unexpected PB input: {} {}".format(input_tensor.name, input_tensor.shape))
    if input_shape[0] == 1 and args.batch_size != 1:
        parser.error("This PB has fixed batch=1; use --batch-size 1")
    if output_tensor.dtype != tf.float32 or output_tensor.shape.as_list() != [input_shape[0], 10]:
        raise ValueError("Unexpected PB output: {} {}".format(output_tensor.name, output_tensor.shape))

    session_config = tf.compat.v1.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True)
    class_counts = np.zeros((len(class_names),), dtype=np.int64)
    class_correct = np.zeros_like(class_counts)
    total_correct = 0
    evaluated = 0
    with tf.compat.v1.Session(graph=graph, config=session_config) as session:
        for start in range(0, len(samples), args.batch_size):
            batch_samples = samples[start : start + args.batch_size]
            batch_inputs = []
            batch_labels = []
            for path, label in batch_samples:
                rgb_uint8 = session.run(decoded_rgb, feed_dict={image_path: path})
                try:
                    normalized = preprocess_rgb_uint8(rgb_uint8)
                except ValueError as error:
                    raise ValueError("{} ({})".format(error, path))
                batch_inputs.append(normalized)
                batch_labels.append(label)

            inputs = np.asarray(batch_inputs, dtype=np.float32)
            logits = session.run(output_tensor, feed_dict={input_tensor: inputs})
            if logits.dtype != np.float32 or logits.shape != (len(batch_samples), 10) or not np.isfinite(logits).all():
                raise RuntimeError("PB returned invalid logits: dtype={}, shape={}".format(logits.dtype, logits.shape))
            predictions = np.argmax(logits, axis=1)
            for label, prediction in zip(batch_labels, predictions):
                class_counts[label] += 1
                class_correct[label] += int(label == prediction)
                total_correct += int(label == prediction)
            evaluated += len(batch_samples)

    total_available = count_images(args.test_dir, class_names)
    print("TensorFlow version: {}".format(tf.__version__))
    print("PB: {}".format(os.path.abspath(args.pb)))
    print("Input/output tensors: {} -> {}".format(input_tensor.name, output_tensor.name))
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
