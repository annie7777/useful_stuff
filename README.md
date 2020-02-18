# useful_stuff

## Agricultural labelling Dataset:

1. Grapevine(instance)

https://zenodo.org/record/3361736

Labelling tool: http://structuralsegm.sourceforge.net/

2. Apple/Pear/Peach flower (pixel)

https://data.nal.usda.gov/dataset/data-multi-species-fruit-flower-detection-using-refined-semantic-segmentation-network

Labelling tool: https://bitbucket.org/phil_dias/freelabel-wacv/src/master/


## NVIDIA with 16.04
### login loop or black screen

DO NOT EDIT grub file with nomodeset only 'Ubuntu' in the start menu

## GCP 
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
