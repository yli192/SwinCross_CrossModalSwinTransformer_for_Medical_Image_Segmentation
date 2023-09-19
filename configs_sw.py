import ml_collections
'''
********************************************************
                   Swin Transformer
********************************************************

img_size (int | tuple(int)): Input image size. Default 224
patch_size (int | tuple(int)): Patch size. Default: 4
in_chans (int): Number of input image channels. Default: 3
num_classes (int): Number of classes for classification head. Default: 1000
embed_dim (int): Patch embedding dimension. Default: 96
depths (tuple(int)): Depth of each Swin Transformer layer.
num_heads (tuple(int)): Number of attention heads in different layers.
window_size (tuple): Window size. Default: 7
mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
drop_rate (float): Dropout rate. Default: 0
attn_drop_rate (float): Attention dropout rate. Default: 0
drop_path_rate (float): Stochastic depth rate. Default: 0.1
norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
patch_norm (bool): If True, add normalization after patch embedding. Default: True
use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
'''
def get_3DSwinNetV0_config():
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 96 #change 128 or 192
    config.depths = (2, 2, 4, 2) #change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (4, 4, 2)
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
    config.seg_head_chan = 16
    config.img_size = (128, 128, 64)
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinNetV01_NoPosEmd_config():
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 128  # change 128 or 192
    config.depths = (2, 2, 10, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (4, 4, 2)
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
    config.seg_head_chan = 16
    config.img_size = (128, 128, 64)
    config.pos_embed_method = 'None'
    return config

def get_3DSwinNetV02_config():
    '''
    Trainable params: 26,914,859

    ML:
    Trainable params: 26,930,417
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 128  # change 128 or 192
    config.depths = (2, 2, 6, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (4, 4, 2)
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
    config.seg_head_chan = 16
    config.img_size = (128, 128, 64)
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinNetV03_config():
    '''
    Trainable params: 26,914,859
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 128  # change 128 or 192
    config.depths = (2, 2, 10, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (4, 4, 2)
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
    config.seg_head_chan = 16
    config.img_size = (128, 128, 64)
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinNetV04_config():
    '''
    Trainable params: 20,621,243
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 192  # change 128 or 192
    config.depths = (2, 2, 4, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (4, 4, 2)
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
    config.seg_head_chan = 16
    config.img_size = (128, 128, 64)
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinNetV05_config():
    '''
    Trainable params: 20,621,243
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 192  # change 128 or 192
    config.depths = (2, 2, 6, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (4, 4, 2)
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
    config.seg_head_chan = 16
    config.img_size = (128, 128, 64)
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinNetV06_config():
    '''
    Trainable params: 20,621,243
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 192  # change 128 or 192
    config.depths = (2, 2, 10, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (4, 4, 2)
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
    config.seg_head_chan = 16
    config.img_size = (128, 128, 64)
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinNet_hecktor2021_V01_config():
    '''
    Trainable params: 20,621,243
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 128  # change 128 or 192
    config.depths = (2, 2, 6, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 5, 5)
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
    config.img_size = (160, 160, 160)
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinNet_hecktor2021_V02_config():
    '''
    Trainable params: 20,621,243
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 128  # change 128 or 192
    config.depths = (2, 2, 10, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 5, 5)
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
    config.img_size = (160, 160, 160)
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinNet_hecktor2021_V03_config():
    '''
    Trainable params: 20,621,243
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 192  # change 128 or 192
    config.depths = (2, 2, 10, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 5, 5)
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
    config.seg_head_chan = 16
    config.img_size = (160, 160, 160)
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinNetNoPosEmbd_config():
    '''
    No position embedding

    Computational complexity:       686.76 GMac
    Number of parameters:           46.77 M
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 128  # change 128 or 192
    config.depths = (2, 2, 6, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 5, 5)
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
    config.seg_head_chan = 16
    config.img_size = (160, 160, 160)
    config.pos_embed_method = 'None'
    return config

def get_3DSwinNetV3_config():
    '''
    Trainable params: 26,914,859
    ML: 19,002,137
    Relaive Sin: 26,914,859
    '''
    config = ml_collections.ConfigDict()
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 112
    config.depths = (2, 2, 2)
    config.num_heads = (4, 4, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.pos_embed_method = 'relative'
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DSwinNetV4_config():
    '''
    Version 4 same as version 2 but no conv skip
    Trainable params: 26,914,859
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = False
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 112
    config.depths = (2, 4, 4)
    config.num_heads = (4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.pos_embed_method = 'relative'
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DSwinNetV4Rot_config():
    '''
    Version 4 same as version 2 but no conv skip
    Trainable params: 26,914,859
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = False
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 112
    config.depths = (2, 4, 4)
    config.num_heads = (4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.pos_embed_method = 'rotary'
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DSwinNetV4NoPosEmbed_config():
    '''
    Version 4 same as version 2 but no conv skip
    Trainable params: 26,914,859
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = False
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 112
    config.depths = (2, 4, 4)
    config.num_heads = (4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.pos_embed_method = 'None'
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DSwinNetV5_config():
    '''
    Version 5 same as version 2 but no conv skip and no trans skip
    Trainable params: 26,914,859
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = False
    config.if_convskip = False
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 112
    config.depths = (2, 4, 4)
    config.num_heads = (4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.pos_embed_method = 'relative'
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DSwinNetV5NoPosEmbed_config():
    '''
    Version 6 same as version 2 but no conv skip and no trans skip
    Trainable params: 26,914,859
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = False
    config.if_convskip = False
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 112
    config.depths = (2, 4, 4)
    config.num_heads = (4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.pos_embed_method = 'None'
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DSwinNetV6_config():
    '''
    Version 6 same as version 2 but no trans skip
    Trainable params: 26,914,859
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = False
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 112
    config.depths = (2, 4, 4)
    config.num_heads = (4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.pos_embed_method = 'relative'
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config

def get_3DSwinNetV6NoPosEmbed_config():
    '''
    Version 6 same as version 2 but no trans skip
    Trainable params: 26,914,859
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = False
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 2
    config.embed_dim = 112
    config.depths = (2, 4, 4)
    config.num_heads = (4, 8, 8)
    config.window_size = (5, 6, 7)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.pos_embed_method = 'None'
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 224)
    return config


def get_3DSwinUNETR_hecktor2021_V01_config():
    '''
    Trainable params: 20,621,243
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 48  # change 128 or 192
    config.depths = (2, 2, 2, 2)  # change 4 to 6,10
    config.num_heads = (3, 6, 12, 24)
    config.window_size = (7, 7, 7)
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
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinUNETR_CMFF_hecktor2021_V01_config():
    '''
    Trainable params: 84014420; best val dice = 0.713
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 48  # change 128 or 192
    config.depths = (2, 2, 2, 2)  # change 4 to 6,10
    config.num_heads = (3, 6, 12, 24)
    config.window_size = (7, 7, 7)
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
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinUNETR_hecktor2021_V01_config():
    '''
    Trainable params: 20,621,243
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 48  # change 128 or 192
    config.depths = (2, 2, 2, 2)  # change 4 to 6,10
    config.num_heads = (3, 6, 12, 24)
    config.window_size = (7, 7, 7)
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
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinUNETR_hecktor2021_V02_config():
    '''
    Trainable params: 20,621,243
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 48  # change 128 or 192
    config.depths = (2, 2, 2, 2)  # change 4 to 6,10
    config.num_heads = (3, 6, 12, 24)
    config.window_size = (7, 7, 7)
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
    config.img_size = (64, 64, 64)
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinUNETR_CMFF_hecktor2021_V02_config():
    '''
    Trainable params: 84014420; best val dice = 0.713
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 30  # change 128 or 192
    config.depths = (2, 2, 2, 2)  # change 4 to 6,10
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
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinUNETR_CMFF_hecktor2021_V03_config():
    '''
    Trainable params: 84014420; best val dice = 0.713
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 48  # change 128 or 192
    config.depths = (2, 2, 2, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
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
    config.pos_embed_method = 'relative'
    return config

def get_3DSwinUNETR_CMFF_hecktor2021_V04_config():
    '''
    Trainable params: 84014420; best val dice = 0.713
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 36  # change 128 or 192
    config.depths = (2, 2, 2, 2)  # change 4 to 6,10
    config.num_heads = (4, 4, 8, 8)
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
    config.pos_embed_method = 'relative'
    return config


def get_3DSwinUNETR_CMFF_hecktor2021_V05_config():
    '''
    Trainable params: 84014420; best val dice = 0.713
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 48  # change 128 or 192
    config.depths = (2, 2, 2, 2)  # change 4 to 6,10
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
    config.pos_embed_method = 'relative'
    return config


def get_3DSwinUNETR_CMFF_hecktor2021_V06_config():
    '''
    Trainable params: 84014420; best val dice = 0.713
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 48  # change 128 or 192
    config.depths = (2, 4, 2, 2)  # change 4 to 6,10
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
    config.pos_embed_method = 'relative'
    return config