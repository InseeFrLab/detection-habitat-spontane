import torch

# with open("../config.yml") as f:
#     config = yaml.load(f, Loader=SafeLoader)

# list_data_dir = download_data(config)
# list_output_dir = prepare_data(config, list_data_dir)
# #download_prepare_test(config)
# model = instantiate_model(config)
# train_dl, valid_dl, test_dl = instantiate_dataloader(
#     config, list_output_dir
# )


def calculate_IOU(output, labels):
    """
    Calculate Intersection Over Union indicator
    based on output segmentation mask of a model
    and the true segmentations mask

    Args:
        output: the output of the segmentation
        label: the true segmentation mask

    """
    preds = torch.argmax(output, axis=1)

    numIOU = torch.sum((preds * labels), axis=[1, 2])  # vaut 1 si les 2 = 1
    denomIOU = torch.sum(torch.clamp(preds + labels, max=1), axis=[1, 2])

    IOU = numIOU / denomIOU
    IOU = [1 if torch.isnan(x) else x for x in IOU]
    IOU = torch.tensor(IOU, dtype=torch.float)
    IOU = torch.mean(IOU)

    return IOU


# calculate num and denomionateur IOU
def calculate_pourcentage_loss(output, labels):
    """
    Calculate the pourcentage of wrong predicted classes
    based on output classification predictions of a model
    and the true classes.

    Args:
        output: the output of the classification
        labels: the true classes

    """
    probability_class_1 = output[:, 1]

    # Set a threshold for class prediction
    threshold = 0.51

    # Make predictions based on the threshold
    predictions = torch.where(probability_class_1 > threshold, torch.tensor([1]), torch.tensor([0]))

    predicted_classes = predictions.type(torch.float)

    misclassified_percentage = (predicted_classes != labels).float().mean()

    return misclassified_percentage


def proportion_ones(labels):
    """
    Calculate the proportion of ones in the validation dataloader.

    Args:
        labels: the true classes

    """

    # Count the number of zeros
    num_zeros = int(torch.sum(labels == 0))

    # Count the number of ones
    num_ones = int(torch.sum(labels == 1))

    prop_ones = num_ones / (num_zeros + num_ones)

    # Rounded to two digits after the decimal point
    prop_ones = round(prop_ones, 2)

    return prop_ones
