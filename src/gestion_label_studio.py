import os
import zipfile
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np


def create_mask_from_label_studio_export(
    num_task,
    name_output,
    type_label="Petites habitations",
    emplacement_zip="../labelstudio.zip",
):
    """
    Extracts binary masks for a single type label from a Label Studio
    export file, combines them into a single mask,
    and saves it as a numpy file.

    Args:
        num_task (str):
            The task number to select masks from.
        name_output (str):
            The file path to save the output numpy file.
            Enter the same name as the image where the mask comes from.
        type_label (str):
            The type of label to select masks for.
            Default is 'Petites habitations'.
        emplacement_zip (str):
            The file path of a zipped Label Studio export file.
            Default is '../labelstudio.zip'.

    Returns:
        Plot of the mask.

    Example:
    >>> create_mask_from_label_studio_export(
            num_task = "1",
            name_output = "mayotte2020nomfichier"
        )

    """
    with zipfile.ZipFile(emplacement_zip, "r") as zip_ref:
        zip_ref.extractall("labelstudio")

    dir = "labelstudio/"

    liste_name = os.listdir(dir)

    list_num_task = [file.split("-")[1] for file in liste_name]
    list_type_label = [file.split("-")[7] for file in liste_name]

    booleen = [
        nt == num_task and tl == type_label
        for nt, tl in zip(list_num_task, list_type_label)
    ]

    list_name_select = [name for name, b in zip(liste_name, booleen) if b]
    list_path_mask = [dir + name for name in list_name_select]

    list_mask = [np.load(path) for path in list_path_mask]
    mask = reduce(lambda x, y: x + y, list_mask)

    mask[mask != 0] = 1

    # Plot the mask using imshow
    plt.imshow(mask, cmap="gray", interpolation="nearest")

    # Show the plot
    plt.show()

    np.save(name_output, mask)
