###Environment and dependencies: *Python 3.6.10 conda 4.7.12 cuda 10.1 Keras 2.3.1 tensorflow 2.3.0 numpy 1.16.2*  
#It is possible that with higher versions data cannot be created properly
# Curriculum Learning
## Data preparation for stage1
Use crop_ips.ipynb or crop_melanoma.ipynb .  
In ./crop/dataset_(1~5)/train/ ./crop/dataset_(1~5)/val/ data will be stored in numpy format.  

## Training for stage1
Set all necessary parameter values in run.py and train2.py and execute as in the following example/   
Example : python run.py --mode　train  

After execution, the model weights will be in the following directory.
Example : ./ips/unet_patch/unet/size_160/dataset_1/weight/  

## Test for stage1
Use uncertainty_ips.ipynb or uncertainty_melanoma.ipynb to make predicted image, label map, uncertainty map and correctness map.  
Example for predicted image : *./ips/unet_patch/unet/size_160/dataset_1/test/*   
Example for label map : *./ips/unet_patch/unet/size_160/dataset_1/label/*  
Example for uncertainty map : *./ips/unet_patch/unet/size_160/dataset_1/test_un/uncertainty_T_0.5/*  
Example for correctness map : *./ips/unet_patch/unet/size_160/dataset_1/test_un/correctness/*  

## Evaluation for stage1
For segmentation evaluation use *$ python run.py --mode　evaluation* 
For uncertainty evaluation use *evaluate_uncertainty.ipynb*
The 5-fold cross-validation results will be in *./ips/unet_patch/unet/size_160/*　
Results for individual folds will be in *./ips/unet_patch/unet/size_160/dataset_1~5*   
seg : *image_evaluate.txt* and unc : *image_evaluate2.txt*

## Data preparation for stage2
Before using crop_ips.ipynb or crop_melanoma.ipynb you need to delete the data in *./crop/dataset_1/train/*  
No need to delete data in *./crop/dataset_1/val/*

## Training for stage2
Execute run.py and train2.py by setting to stage2.  　 
Example : *python run.py --mode　train*  
After execution, the model weights will be in the following directory : *./ips/unet_patch/unet/size_160/dataset_1/stage2/weight/*
and so on, as explained above.

# Training with Uncertainty Loss (Method 2 in the paper)
## Data preparation
Use crop_ips.ipynb or crop_melanoma.ipynb  
In ./crop/dataset_1/train/ ./crop/dataset_1/val/ data will be stored in numpy format.  
For training & test with entropy_loss_method_ips.ipynb or entropy_loss_method_melanoma.ipynb you need to restart the notebook between training and test (for melanoma training does not seem to go well). 

## Evaluation
For segmentation evaluation use *python run.py --mode　evaluation*  
For uncertainty evaluation use *evaluate_uncertainty.ipynb*  
The 5-fold cross-validation results will be *in ./ips/unet_patch/unet/size_160/*  
Results for individual folds will be in *./ips/unet_patch/unet/size_160/dataset_1~5 * 
seg : *image_evaluate.txt*　and　unc : *image_evaluate2.txt* 
#　Directory structure and order: All code is inside program. When executing code make sure code and data folders are in the same path. 
program -> README.md (this document)  -> crop(make data) -> model(BayesianUNet[keras_bcnn], <other model>) -> evaluation(segmentation, uncertainty) -> curriculum(method1　curriculum learning) -> entropy_loss(method2　Uncertainty loss)
