import os
import gin
import glob
import tqdm
import pickle
import argparse
import numpy as np
import tensorflow as tf
import gin.tf.external_configurables

import models
import transnet
import training
import metrics_utils
import create_dataset
import input_processing
import visualization_utils
import transnetv2

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_batches(frames):
    reminder = 50 - len(frames) % 50
    if reminder == 50:
        reminder = 0
    frames = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25), 0)

    def func():
        for i in range(0, len(frames) - 50, 50):
            yield frames[i:i+100]
    return func()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="+", help="path to video files to process")
    parser.add_argument("--weights", type=str, default=None,
                        help="path to TransNet V2 weights, tries to infer the location if not specified")
    parser.add_argument('--visualize', action="store_true",
                        help="save a png file with prediction visualization for each extracted video")
    args = parser.parse_args()

    model = transnet.TransNetV2(args.weights)
    for file in args.files:
        if os.path.exists(file + ".predictions.txt") or os.path.exists(file + ".scenes.txt"):
            print(f"[TransNetV2] {file}.predictions.txt or {file}.scenes.txt already exists. "
                  f"Skipping video {file}.", file=sys.stderr)
            continue

        video_frames, single_frame_predictions, all_frame_predictions = \
            model.predict_video(file)

        predictions = np.stack([single_frame_predictions, all_frame_predictions], 1)
        np.savetxt(file + ".predictions.txt", predictions, fmt="%.6f")

        scenes = model.predictions_to_scenes(single_frame_predictions)
        np.savetxt(file + ".scenes.txt", scenes, fmt="%d")

        if args.visualize:
            if os.path.exists(file + ".vis.png"):
                print(f"[TransNetV2] {file}.vis.png already exists. "
                      f"Skipping visualization of video {file}.", file=sys.stderr)
                continue

            pil_image = model.visualize_predictions(
                video_frames, predictions=(single_frame_predictions, all_frame_predictions))
            pil_image.save(file + ".vis.png")



    def predict_raw(self, frames: np.ndarray):
        assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
            "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
        frames = tf.cast(frames, tf.float32)

        logits, dict_ = self._model(frames)
        single_frame_pred = tf.sigmoid(logits)
        all_frames_pred = tf.sigmoid(dict_["many_hot"])

        return single_frame_pred, all_frames_pred

    net(tf.zeros([1] + options["input_shape"], tf.float32))
    net.load_weights(os.path.join(args.log_dir, "weights-{:d}.h5".format(args.epoch)))
    files = glob.glob(os.path.join(args.directory, "*.npy"))

    results = []
    total_stats = {"tp": 0, "fp": 0, "fn": 0}

    dataset_name = [i for i in args.directory.split("/") if i != ""][-1]
    img_dir = os.path.join(args.log_dir, "results", "{}-epoch{:d}".format(dataset_name, args.epoch))
    os.makedirs(img_dir, exist_ok=True)

    for np_fn in tqdm.tqdm(files):
        predictions = []
        frames = np.load(np_fn)

        for batch in get_batches(frames):
            one_hot = predict_raw(batch)
            predictions.append(one_hot[25:75])

        predictions = np.concatenate(predictions, 0)[:len(frames)]
        gt_scenes = np.loadtxt(np_fn[:-3] + "txt", dtype=np.int32, ndmin=2)

        _, _, _, (tp, fp, fn), fp_mistakes, fn_mistakes = metrics_utils.evaluate_scenes(
            gt_scenes, metrics_utils.predictions_to_scenes((predictions >= args.thr).astype(np.uint8)),
            return_mistakes=True)

        total_stats["tp"] += tp
        total_stats["fp"] += fp
        total_stats["fn"] += fn

        if len(fp_mistakes) > 0 or len(fn_mistakes) > 0:
            img = visualization_utils.visualize_errors(
                frames, predictions,
                create_dataset.scenes2zero_one_representation(gt_scenes, len(frames))[1],
                fp_mistakes, fn_mistakes)
            if img is not None:
                img.save(os.path.join(img_dir, os.path.basename(np_fn[:-3]) + "png"))

        results.append((np_fn, predictions, gt_scenes))

    with open(os.path.join(args.log_dir, "results", "{}-epoch{:d}.pickle".format(dataset_name, args.epoch)), "wb") as f:
        pickle.dump(results, f)

    p = total_stats["tp"] / (total_stats["tp"] + total_stats["fp"])
    r = total_stats["tp"] / (total_stats["tp"] + total_stats["fn"])
    f1 = (p * r * 2) / (p + r)
    print(f"""
    Precision:{p*100:5.2f}%
    Recall:   {r*100:5.2f}%
    F1 Score: {f1*100:5.2f}%
    """)