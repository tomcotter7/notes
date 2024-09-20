# AWS

## Load Balancer -> EC2 Instance
Find documentation [here](https://docs.aws.amazon.com/elasticloadbalancing/latest/classic/elb-getting-started.html) on getting a load balancer to point to an EC2 instance.

## S3

`aws s3 sync` can sync all the files in a directory to an S3 bucket. For example, to sync all the files in the current directory to an S3 bucket called `my-bucket`, you can run:

```bash

aws s3 sync . s3://my-bucket

```
## Sagemaker

### Deploying ONNX Optimized Models

We can deploy ONNX optimised models to an NVIDIA Triton server to perform inference. Note that with TensorRT ONNX models on Triton we need to compile the model on the same type of hardware that we deploy it with, thus we use the same GPU instance that we will be using for deployment of our SageMaker Real-Time Endpoint later. An issue with the Triton server is that we need the `config.pbtxt` file, which expects the Input/Output shapes needed for our model.

An example config file is as follows:
```json
    name: "sentence_onnx"
    platform: "onnxruntime_onnx"
    input: [
        {
            name: "input_ids"
            data_type: TYPE_INT64
            dims: [512]
        },

        {
            name: "attention_mask"
            data_type: TYPE_INT64
            dims: [512]
        }
    ]
    output [
      {
        name: "last_hidden_state"
        data_type: TYPE_FP32
        dims: [ -1, -1, 768 ]
      }
    ]
    instance_group {
      count: 1
      kind: KIND_GPU
    }
    dynamic_batching {
        max_queue_delay_microseconds: 1000
        preferred_batch_size: 5
    }
```

So this would be a model that takes in a vector input of size 512, and outputs a vector of size 768. The `instance_group` specifies that we want to use a GPU, and the `dynamic_batching` specifies that we want to batch the requests in groups of 5.

[This article](https://aws.plainenglish.io/deploying-transformers-onnx-models-on-amazon-sagemaker-7689e8710328) gives some more details on how to deploy with Triton. See the notebook [here](https://github.com/RamVegiraju/SageMaker-Deployment/blob/master/RealTime/Multi-Model-Endpoint/Triton-MME-GPU/triton-mme-onnx-embeddings.ipynb?source=post_page-----7689e8710328--------------------------------)
