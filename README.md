# GoogleCloudML
## Introduction
This repository contains different simple examples of ML architectures which run on GoogleCloudML. The files 4 to 6 contain examples of different deep learning architectures ready for the cloud. The bigger ones have been adapted to run on a single GPU model (standard_gpu), which had a 11 GB RAM limit (as of July 2017).

I created this examples for myself because the Google Tutorials seemed quite complex and distributed over a lot of pages. I nevertheless recommend to do the basic setup as described [here](https://cloud.google.com/ml-engine/docs/quickstarts/command-line). It helps you set up your environment, I highly recommend using the Google Shell. Moreover create a [project](https://console.cloud.google.com/cloud-resource-manager) and a bucket associated to it if you intend to use external files.
If you feel like doing an complete example from Google, do it [here](https://cloud.google.com/ml-engine/docs/how-tos/getting-started-training-prediction). However for me too many details were hidden to get a full understanding of the mechanism.

Before running anything on the cloud, check if the task runs locally by running the following command in the root directory of the project:
```bash
python -m trainer/4-convnet
```

## Running a job
On your Google cloud shell, make sure you have tensorflow and other libraries which you might require installed (scipy does not run on the shell due to resource constraints). Other libraries have to be installed similarly, the usual `pip install package` does not work. 

```bash
pip download tensorflow
pip install --user -U tensorflow*.whl
```

Create the recommended structure, with an input/ and trainer/ folder as here on github. And copy the files from the bucket to the local machine:
```bash
gsutil cp gs://applicationfromscratchbucket/file_to_copy .
```
The *copy_files* script contains a script which did just that for my example.

Next create the environmental variables, it is mandatory to choose a unique *JOB_NAME* each time you submit a job, hence you should re-run the first two lines before each job. It will also help you identify the output folders.

```bash
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="convnet_$now"
STAGING_BUCKET=gs://applicationfromscratchbucket
REGION="europe-west1"
INPUT_PATH=${STAGING_BUCKET}/input
PACKAGE_PATH=trainer
OUTPUT_PATH=$STAGING_BUCKET/$JOB_NAME
```

Now we are ready to launch a job. The yaml contains information which type of cloud server we want. The file in this repository asks for a *standard_gpu* machine, which has one GPU. Be aware that not every region has GPU machines available.
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
Note that the lonley *--* indicate that the arguments for the python program follow, everything before is for gcloud. The results of the task will be displayed in the logs of Google Cloud.
