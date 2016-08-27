import numpy as np


def mask_to_category(mask):
    """
    """
    
    assert (len(mask.shape) == 4)
    assert (mask.shape[1] == 1)
    
    im_h = mask.shape[2]
    im_w = mask.shape[3]
    
    mask = np.reshape(mask,(mask.shape[0], im_h * im_w))
    new_mask = np.empty((mask.shape[0], im_h * im_w, 2))
    
    for i in range(mask.shape[0]):
        for j in range(im_h * im_w):
            
            if  mask[i,j] == 0:
                new_mask[i,j,0] = 1
                new_mask[i,j,1] = 0
            else:
                new_mask[i,j,0] = 0
                new_mask[i,j,1] = 1
                
    return new_mask