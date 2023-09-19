# Model Overview
This repository contains the code for SwinCross: Cross-modal Swin Transformer for 3D Medical Image Segmentation. 

### Installing Dependencies
Dependencies can be installed using:
``` bash
pip install -r requirements.txt
```

### Training

A SwinCross network with standard hyper-parameters for the task of head and neck tumor semantic segmentation (HECTOR dataset) can be defined as follows:

``` bash
model = SwinCross(
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 48 
    config.depths = (2, 4, 2, 2)  
    config.num_heads = (3, 6, 12, 24)
    config.window_size = (3, 3, 3)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = True
    config.spe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.seg_head_chan = config.embed_dim // 2
    config.img_size = (96, 96, 96)
    config.pos_embed_method = 'relative')
```

Note that you need to provide the location of your dataset directory by using ```--data_dir```.

To initiate distributed multi-gpu training, ```--distributed``` needs to be added to the training command.

To disable AMP, ```--noamp``` needs to be added to the training command.


### Testing
You can use the state-of-the-art pre-trained TorchScript model or checkpoint of UNETR to test it on your own data.

Once the pretrained weights are downloaded, using the links above, please place the TorchScript model in the following directory or 
use ```--pretrained_dir``` to provide the address of where the model is placed:

```./pretrained_models``` 

The following command runs inference using the provided checkpoint:
``` bash
python test.py
--infer_overlap=0.5
--data_dir=/dataset/dataset0/
--pretrained_dir='./pretrained_models/'
--saved_checkpoint=ckpt
``` 

Note that ```--infer_overlap``` determines the overlap between the sliding window patches. A higher value typically results in more accurate segmentation outputs but with the cost of longer inference time.

If you would like to use the pretrained TorchScript model, ```--saved_checkpoint=torchscript``` should be used.


