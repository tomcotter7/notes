# AWS

## Load Balancer -> EC2 Instance
Find documentation [here](https://docs.aws.amazon.com/elasticloadbalancing/latest/classic/elb-getting-started.html) on getting a load balancer to point to an EC2 instance.

## S3

`aws s3 sync` can sync all the files in a directory to an S3 bucket. For example, to sync all the files in the current directory to an S3 bucket called `my-bucket`, you can run:

```bash

aws s3 sync . s3://my-bucket

```

## Test

Hello
