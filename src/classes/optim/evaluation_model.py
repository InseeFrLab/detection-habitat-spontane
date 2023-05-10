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
def evaluer_modele_sur_jeu_de_test_segmentation_pleiade(test_dl, model, tile_size, batch_size, mlflow = False):

    npatch = int((2000/tile_size)**2)
    nbatchforfullimage = int(npatch/batch_size)
    
    if not npatch % nbatchforfullimage == 0:
        print("Le nombre de patchs n'est pas divisible par la taille d'un batch")
        return None 
    
    list_labeled_satellite_image = []

    for idx, batch in enumerate(test_dl):
        
        images, label, dic = batch
        output_model = model(images)
        mask_pred = np.array(torch.argmax(output_model, axis  = 1))

        for i in range(batch_size):    
            pthimg = dic["pathimage"][i]
            si = SatelliteImage.from_raster(
                file_path = pthimg,
                dep = None,
                date = None,
                n_bands= 3
            )
            si.normalize()
        
            list_labeled_satellite_image.append( 
                SegmentationLabeledSatelliteImage(
                    satellite_image =  si ,
                    label= mask_pred[i],
                    source= "",
                    labeling_date = ""
                )
            )
        

        if ((idx+1) % nbatchforfullimage) == 0:
            print("ecriture image")
            if not os.path.exists("img/"):
                os.makedirs("img/")

            fig1 = plot_list_segmentation_labeled_satellite_image(list_labeled_satellite_image,[0,1,2])
    
            filename = pthimg.split('/')[-1]
            filename = filename.split('.')[0]
            filename = '_'.join(filename.split('_')[0:6])
            plot_file = filename + ".png"
        
            fig1.savefig(plot_file)
            list_labeled_satellite_image = []
            
            if mlflow:
                mlflow.log_artifact(plot_file, artifact_path="plots")
            
        del images, label, dic
    


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