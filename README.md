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
gcsfuse --background --threads apple-data[gcp folder] bucket/[folder on instance]
```

### Copy vm folders to bucket
```
gsutil -m cp -r dir-you-want-copy-from gs://my-test-apple-data/deeplab_results/
```

### Copy local folder to vm machine

```
gcloud compute scp --recurse [LOCAL_FILE_PATH] [INSTANCE_NAME]:/home/user/... --zone [asia....]
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

