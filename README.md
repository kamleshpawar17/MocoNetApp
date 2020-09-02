## Installation
clone the repository and run the setup.sh file
````
sh setup.sh
````

## Run
To start GUI run
````
python runMcAPP.py
````
In web browser go to [http://127.0.0.1:5000](http://127.0.0.1:5000) (tested with Google Chrome)


This should display GUI

![GuiScreenshot](./static/ScreenShot.png)

Click on **Choose Files** and select all the dicom files (*.dcm) for the scan, then click on **upload**, which will display the motion degraded image in the left viewer

Click on **Process** to run the AI motion correction algorithms on the input images.

Click in **Download** to save motion-corrected images. (It will be a zip file containing all the processed dicom images)

The GUI uses [papya viewer](https://github.com/rii-mango/Papaya).
Explore the top right corner icons of the image viewer for image rotation and dynamic range settings. 

**Note:** If the viewer shows the same image even after uploading the new images, this is because the browser is displaying images from the cache. On macOS + google chrome, use cmd+shift+r to bypass cache or use shift+reload-click 

## Dependencies
Following dependencies must be installed for the GUI to work

nibabel, gevent, flask, progressbar, nibabel, pydicom, Keras

### Alternatively, a self contained docker container is available [here](https://hub.docker.com/r/kamleshp/moconetapp) 
