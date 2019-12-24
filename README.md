# Multimodal Brain Tumor Segmentation 

## About Data 
The data was collected from Multimodal Brain Tumor Segmentation Challenge 2018 (BraTS) Data. The link to access the data:- https://www.med.upenn.edu/sbia/brats2018/data.html

### Imaging Data Description
1. All BraTS multimodal scans were available as NIfTI files (.nii.gz) having different modalitied:-
  - Native (T1) 
  - Post-contrast T1-weighted (T1Gd)
  - T2-weighted (T2) 
  - T2 Fluid Attenuated Inversion Recovery (FLAIR)

2. The three segmentation Labels as described in the BraTS reference paper, published in IEEE Transactions for Medical Imaging (https://ieeexplore.ieee.org/document/6975210):- 
  - GD-enhancing tumor (ET — label 4) 
  - Peritumoral edema (ED — label 2)
  - Necrotic and non-enhancing tumor core (NCR/NET — label 1)
  - Remaining Region (label 0)
3. The data were distributed after their pre-processing, i.e. co-registered to the same anatomical template, interpolated to the same resolution (1 mm^3) and skull-stripped.

### Task
Segmentation of gliomas in pre-operative MRI scans. Use the provided clinically-acquired training data to produce segmentation labels.

## Methodology
### Data Pre-processing
- As, data was already skull-stripped, then mri each patients mri scans volume was collected and combined to form an numpy array of size (N,S,N1,N1,X).
Here N = Number of HGG/LGG data
     S = Number of total 2D slices correspond to each mri 3D volume imaginary.
     N1 = Dimension of each 2D slice
     X = Number of modalities
- So, here as Google Colab Free GPU was used to do all pre-processing and training so due to RAM consumption the data was handeled appropriately:-
1. Firstly, HGG and LGG folders were handeled one by one to extract the 3D volume of (N,155,240,240,4) dimension. Now, HGG contains 210 patients' scans and LGG contains 75 patients' scans so, the HGG data was divided in three sets namely data11.npy, data12.npy, data13.npy each consisting of 70 patients thus N=70 and data2.npy correspond to LGG data with N=75. Similarly, corresponding ground truth were also extracted with dimension of (N,155,240,240) namely gt11.npy, gt12.npy, gt13.npy, gt2.npy respectively.
2. Now, each data was pre-processed one by one. Now, as all 155 slices does not show tumor region so here only mid portion i.e. from 30th slice to 120th slice was taken for creating the final data, and finally all were reshaped to (N1,240,240,4) for data and (N1,240,240,4) for ground truth using one-hot encoding. Here, N1 = 90X70 = 5600 for HGG, N1 = 90X75 = 6750 for LGG.
3. Next, each data is cropped to centre with final dimension of (N1,192,192,4).
4. Finally, the data was randomly split into training, validation and test data with 60%:20%:20% of ratio respectively.
### Proposed Model
Here we have proposed U-Net for our semnatic segmentation problem:-
![](/unet.png)
### Dice Coefficient & Dice Coefficient Loss Function
- Wikipedia:- Sørensen's original formula was intended to be applied to discrete data. Given two sets, X and Y, it is defined as:-

     ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/a80a97215e1afc0b222e604af1b2099dc9363d3b)

- Here |X| and |Y| are the cardinalities of the two sets (i.e. the number of elements in each set). The Sørensen index equals   twice the number of elements common to both sets divided by the sum of the number of elements in each set. 
- For our metric, small modification was made in denominator side i.e. instead of sum of absolute of X and Y sum of square of X and Y was taken.Here, X = y_true, Y = y_pred. 
  - The implemeted python code for dice coefficient:-
  ```python
  def dice_coef(y_true, y_pred, epsilon=1e-6):
      intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
      return (2. * intersection) / (K.sum(K.square(y_true),axis=-1) + K.sum(K.square(y_pred),axis=-1) + epsilon)
  ```
  - In order to formulate a loss function which can be minimized, we'll simply use 1−dice_coef:- 
  ```python
  def dice_coef_loss(y_true, y_pred):
      return 1-dice_coef(y_true, y_pred)
  ```
## Results Obtained
- HGG Result Samples

  >![](https://github.com/as791/Brain-Tumor-Segmentation-BRaTS-18/blob/master/Result%20Samples/HGG-1.png)
  >![](https://github.com/as791/Brain-Tumor-Segmentation-BRaTS-18/blob/master/Result%20Samples/HGG-2.png)
  >![](https://github.com/as791/Brain-Tumor-Segmentation-BRaTS-18/blob/master/Result%20Samples/HGG-1.png)
  
- LGG Result Samples

  >![](https://github.com/as791/Brain-Tumor-Segmentation-BRaTS-18/blob/master/Result%20Samples/LGG-1.png)
  >![](https://github.com/as791/Brain-Tumor-Segmentation-BRaTS-18/blob/master/Result%20Samples/LGG-2.png)
  
## Evaluated Results
| Test Data|Dice Coefficient| 
| ------------- |:-------------:| 
| HGG Set-1   |   0.9795   |
| HGG Set-2   |   0.9855   |
| HGG Set-3   |   0.9793   |
| LGG         |   0.9950   |
