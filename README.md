# Multitask computer aided diagnosis of breast masses



In this page, a multitask computer aided diagnosis (CADx) framework is devised to investigate interpretability for classifying breast masses. We investigate interpretability in CADx with the proposed multitask CADx framework. 



![image_2018-11-28 23_07_21](https://user-images.githubusercontent.com/44894722/49157188-8942c880-f362-11e8-9719-e73f15ab3fde.png)




## Dependencies
* MATLAB2014
* tensorflow >= 0.10.0rc0
* h5py 2.3.1
* scipy 0.18.0
* Pillow 3.1.0
* opencv2


## How to use
First, go into the 'data' directory and download "DDSM dataset"

For data consistency, mammograms scanned by Howtek 960 were selected from the DDSM dataset.
By using LJPEG2PNG tool "DDSMUtility", load breast image.
Preprocess the DDSM dataset to create a h5 file. It resizes images to 64*64.
   


Finally, go into the 'code' directory and run 'main_ICADx.py'.
    
    cd ../code
    python main_ICADx.py
    

You can see the samples in 'img and test' directory.


## Reference Code
https://github.com/trane293/DDSMUtility
