import zipfile
import os 
import numpy as np
import matplotlib.pyplot as plt

def creer_mask_from_label_studio_export(num_task,type_label,emplacement_zip,name_output):
    with zipfile.ZipFile(emplacement_zip, 'r') as zip_ref:
        zip_ref.extractall('labelstudio')
    
    dir  = 'labelstudio/'

    liste_name = os.listdir(dir)

    list_num_task = [file.split("-")[1] for file in liste_name]
    list_type_label =[file.split("-")[7] for file in liste_name]

    booleen = [nt == num_task and tl == type_label for nt, tl in zip(list_num_task,list_type_label)]

    list_name_select = [ name for name, b in zip(liste_name,booleen) if b]
    list_path_mask  = [ dir +  name for name in list_name_select]


    # lecture d'un seul 

    list_mask = [ np.load(path) for path in list_path_mask]
    from functools import reduce
    mask = reduce(lambda x, y: x + y, list_mask)

    mask[mask != 0] = 1

    # Plot the mask using imshow
    plt.imshow(mask, cmap='gray', interpolation='nearest')

    # Show the plot
    plt.show()
    
    np.save(name_output,mask)