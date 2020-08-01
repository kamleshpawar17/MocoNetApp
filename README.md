# MocoNetApp

This repository contains a browser based GUI software for the deep learning motion correction algorithm publlished in:

Pawar K, Chen Z, Shah NJ, Egan GF. Suppressing motion artefacts in MRI using an Inception‐ResNet network with motion simulation augmentation. *NMR in Biomedicine. 2019 Dec 22:e4225. doi: [https://doi.org/10.1002/nbm.4225](https://doi.org/10.1002/nbm.4225)* 

# Installation

# Run
To start GUI run
````
python runMcAPP.py
````
In web browser go to [http://127.0.0.1:5000](http://127.0.0.1:5000/process)

This should display GUI as

![GuiScreenshot](https://github.com/kamleshpawar17/MocoNetApp/tree/master/static/ScreenShot.png)

Click on **Choose Files** and select all the dicom files (*.dcm) for the scan, follwed by upload, which will display the motion degraded image in the left viewer

Click on **Process** to run the AI motion correction on the input image.

Click in **Download** to save motion corrected images. (It will be a zip file which contains all the precessed dicom images)