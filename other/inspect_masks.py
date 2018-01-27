
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
 
masks_pred = np.load('masksTestPredicted.npy')
masks_true = np.load('masksTest.npy')
imgs = np.load('imagesTest.npy')
print(imgs.shape)
for i in range(30):
    
    print "predicted mask %d" % i
    fig,ax = plt.subplots(2,3,figsize=[8,8])
    ax[0,0].imshow(imgs[i][:, :, 0],cmap='gray')
    ax[0,1].imshow(masks_true[i][:, :, 0])
    ax[0,2].imshow(imgs[i][:, :, 0]*masks_true[i][:, :, 0],cmap='gray')
    ax[1,0].imshow(imgs[i][:, :, 0],cmap='gray')
    ax[1,1].imshow(masks_pred[i][:, :, 0])
    ax[1,2].imshow(imgs[i][:, :, 0]*masks_pred[i][:, :, 0],cmap='gray')
    plt.show()
    
    

