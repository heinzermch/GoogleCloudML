# GoogleCloudML
## Introduction
This repository contains different examples of ML architectures which run on GoogleCloudML. Especially the files 4 to 6 contain examples of different deep learning architectures in the cloud. The bigger ones have been adapted to run on a single GPU model (standard_gpu), which had a 11 GB RAM limit (as of July 2017).

I created this examples for myself because the Google Tutorials seemed quite complex and distributed over a lot of pages. I nevertheless recommend to do the basic setup as described (here)[https://cloud.google.com/ml-engine/docs/quickstarts/command-line]. It helps you set up your environment, I highly recommend using the Google Shell. Create a (project)[https://console.cloud.google.com/cloud-resource-manager] and a bucket associated to it if you intend to use external files.
If you feel like doing an complete example, do (this)[https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction]. However for me too many details were hidden to get a full understanding of the mechanism.

Before running anything on the cloud, check if the task runs locally by running the following command in the root directory of the project:
```bash
python -m trainer/4-convnet
```


## Running a job
On your Google cloud shell, make sure you have tensor-flow and other libraries which you might require installed (scipy does not run on the shell due to resource constraints). Other libraries have to be installed similarly.

```bash
pip download tensorflow
pip install --user -U tensorflow*.whl
```

Create the recommended structure, with an input/ and trainer/ folder as here on github. And copy the files from the bucket to the local machine:
```bash
gsutil cp gs://applicationfromscratchbucket/file_to_copy .
```

Create the environmental variables, it is mandatory to choose a new *JOB_NAME* each time you submit a job. Also don't forget to adapt the bucket and put the input files into the directory indicated by the *INPUT_PATH*:

```bash
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="scriptednet_$now"
STAGING_BUCKET=gs://applicationfromscratchbucket
REGION="europe-west1"
INPUT_PATH=${STAGING_BUCKET}/input
PACKAGE_PATH=trainer
OUTPUT_PATH=$STAGING_BUCKET/$JOB_NAME
```

Now we are ready to launch a job.
```bash
gcloud ml-engine jobs submit training $JOB_NAME \
        --package-path $PACKAGE_PATH \
        --staging-bucket $STAGING_BUCKET \
        --module-name=trainer.4-convnet \
        --region $REGION \
        --config=trainer/cloudml-gpu.yaml \
	-- --input_dir=$INPUT_PATH \
        --output_dir=$OUTPUT_PATH
```
Note that the two lonely *--* indicate that the arguments for the python program follow, everything before is only for gcloud. The results of the task will be displayed in the logs of Google Cloud.
