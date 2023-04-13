import torch

# Calculate IOU    
def calculate_IOU(output,labels):
    preds = torch.argmax(output,axis = 1)

    numIOU = torch.sum((preds * labels),axis = [1,2]) # vaut 1 quand les 2 valent 1
    denomIOU = torch.sum(torch.clamp(preds+labels,max = 1),axis = [1,2])

    IOU =  numIOU/denomIOU
    IOU= torch.tensor([1 if torch.isnan(x) else x for x in IOU],dtype =torch.float)
    IOU = torch.mean(IOU)
    return(IOU)