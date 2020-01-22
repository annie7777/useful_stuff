# useful_stuff

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
