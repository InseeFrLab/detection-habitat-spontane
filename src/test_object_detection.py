import mlflow
import torch
from run_training_pipeline import download_data, prepare_train_data, prepare_test_data, instantiate_dataloader
from classes.data.satellite_image import SatelliteImage
from classes.data.labeled_satellite_image import DetectionLabeledSatelliteImage
from utils.utils import get_root_path
import yaml
from yaml.loader import SafeLoader
import numpy as np


def decode_predictions(
    prediction,
    score_threshold,
    nms_iou_threshold
):
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]    # Remove any low-score predictions.
    if score_threshold is not None:
        want = scores > score_threshold
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]    # Remove any overlapping bounding boxes using NMS.
    if nms_iou_threshold is not None:
        want = torchvision.ops.nms(
            boxes = boxes,
            scores = scores,
            iou_threshold = nms_iou_threshold)
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]
    
    return (boxes.cpu().detach().numpy(), 
            labels.cpu().detach().numpy(), 
            scores.cpu().detach().numpy())


def main():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # Open the file and load the file
    with open(get_root_path() / "config.yml") as f:
        config = yaml.load(f, Loader=SafeLoader)

    tile_size = config["data"]["tile_size"]
    batch_size_test = config["optim"]["batch_size_test"]
    task_type = config["data"]["task"]
    source_data = config["data"]["source_train"]
    n_bands = config["data"]["n_bands"]
    src_task = source_data + task_type

    list_data_dir, list_masks_cloud_dir, test_dir = download_data(config)

    list_output_dir = prepare_train_data(config, list_data_dir, list_masks_cloud_dir)
    prepare_test_data(config, test_dir)

    train_dl, valid_dl, test_dl = instantiate_dataloader(config, list_output_dir)

    # Load model
    model = mlflow.pytorch.load_model(f"models:/obj-detection/1")

    # Test Evaluation
    model.eval()

    valid_dl_iterator = iter(valid_dl)
    for j in range(100):
        batch = next(valid_dl_iterator)

        images, label, dic = batch

        # Inference
        model = model.to(device)
        images = images.to(device)
        output_model = model(images)

        for i in range(len(batch)):
            prediction = output_model[i]
            boxes, labels, scores = decode_predictions(
                prediction,
                score_threshold=0.8,
                nms_iou_threshold=0.2
            )

            pthimg = dic[i]["pathimage"]
            si = SatelliteImage.from_raster(
                file_path=pthimg, dep=None, date=None, n_bands=n_bands
            )
            si.normalize()

            # Plot pred
            labeled_si = DetectionLabeledSatelliteImage(
                satellite_image=si,
                label=boxes,
                source="",
                labeling_date="",
            )

            fig1 = labeled_si.plot(bands_indices=[0, 1, 2])
            plot_file = f"{j}_{i}.png"
            fig1.savefig(plot_file)

            # Plot GT
            gt_labeled_si = DetectionLabeledSatelliteImage(
                satellite_image=si,
                label=label[i].numpy(),
                source="",
                labeling_date="",
            )

            fig2 = gt_labeled_si.plot(bands_indices=[0, 1, 2])
            plot_file = f"{j}_{i}_gt.png"
            fig2.savefig(plot_file)


if __name__ == "__main__":
    main()