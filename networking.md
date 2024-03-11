# Networking

## Corporate Firewall / ZScaler Issues

ZScaler blocks certain SSL requests. The fix is to downgrade `requests` module to `2.27.1` and set the `CURL_CA_BUNDLE` environment variable to be the empty string. This can be done via `os`. 
 
 The version of this specifically related to `huggingface` is documented [here](https://github.com/huggingface/transformers/issues/25552).


