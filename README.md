# useful_stuff

## Agricultural labelling Dataset:

### 1. Grapevine(instance)

Paper: https://www.sciencedirect.com/science/article/pii/S0168169919315765

Link: https://zenodo.org/record/3361736

Labelling tool: http://structuralsegm.sourceforge.net/

### 2. Apple/Pear/Peach flower (pixel)

Paper: https://ieeexplore.ieee.org/document/8392727

Link: https://data.nal.usda.gov/dataset/data-multi-species-fruit-flower-detection-using-refined-semantic-segmentation-network

Labelling tool: https://bitbucket.org/phil_dias/freelabel-wacv/src/master/

### 3. Apple (BBox), available in different formats(RGB, Depth and range corrected intensity)

Paper: https://www.sciencedirect.com/science/article/pii/S0168169919301413

Link: http://www.grap.udl.cat/en/publications/datasets.html
### 4. Fruit 360 (Apple Varieties Classfication)

Link: https://www.kaggle.com/moltean/fruits/kernels

### 5. Apple Defect detection(244 images of defected apples)

Link: https://github.com/raheelsiddiqi2013/apple-defect-detection

### 6. Synthetic fruit dataset

Link: https://public.roboflow.ai/object-detection/synthetic-fruit/1

## NVIDIA with 16.04
### login loop or black screen

DO NOT EDIT grub file with nomodeset only 'Ubuntu' in the start menu

## GCP
### keep processes running after ending ssh session 

type ```screen``` in ssh

Ctrl+A

type ```python FCN......```

Ctrl+D

Ref: https://medium.com/@arnab.k/how-to-keep-processes-running-after-ending-ssh-session-c836010b26a3

### Mount bucket on vm
```
gcsfuse --background --threads my-test-apple-data[bucket] bucket/[folder on instance]
```

### Copy vm folders to bucket
```
gsutil -m cp -r dir-you-want-copy-from gs://my-test-apple-data/deeplab_results/
```

### Copy local folder to vm machine

```
gcloud compute scp --recurse [LOCAL_FILE_PATH] annie_xu_wang7@[INSTANCE_NAME]:/home/user/... --zone [asia....]
```

### Copy bucket folder to local machine
```
gsutil cp - r gs://... C:/users
```

## Python
### extend file with other format
file.extend(glob.glob(..))


## Github
### Add local to remote repo
```
git remote add origin <remote repository URL>
git remote -v
git push origin master
```
## CUDNN_STATUS_NOT_INITIALIZED

found this is because the pretrained model giving bad performance so currently the solution is training from scratch

## Build opencv4 from source

https://github.com/milq/milq/blob/master/scripts/bash/install-opencv.sh
cmake -DCMAKE_BUILD_TYPE=RELEASE  -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_TIFF=ON -DBUILD_LIBPROTOBUF_FROM_SOURCES=ON -DOPENCV_GENERATE_PKGCONFIG=YES ..

if using conda then remove condapath

## Best cnn visualization video

https://www.youtube.com/watch?v=RNnKtNrsrmg&feature=youtu.be

## create video from images in a folder

cat *.jpg | ffmpeg -f image2pipe -r 1 -vcodec mjpeg -i - -vcodec libx264 out.mp4

## Switch project and zone gcloud 

gcloud config set project apple-sensing

gcloud config set compute/region us-east4

Ref: https://stackoverflow.com/questions/45125143/how-to-change-region-zone-in-google-cloud

## Refresh in linux

sudo apt install gnome-shell
gnome-shell --replace & disown

## YoloV5 RuntimeError: No such operator torchvision:nms
https://github.com/ultralytics/yolov5/issues/352#issuecomment-661842317

