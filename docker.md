# Docker

## Uploading Docker Images to other servers
Save the docker image to a tar file: `docker save -o <path for generated tar file> <image name>`
Copy the tar file to the other computer and load it: `docker load -i <path to image tar file>`
Run the compose file: `docker-compose up -d`, this should be run in the same directory as the compose file.
This is mainly useful when you have a computer without internet connection that can't download the images / images aren't available on Docker Hub.


