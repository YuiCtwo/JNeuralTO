optim = dict(
    material=dict(
        type='Adam',
        lr=2e-4,
        eps=1e-15,
        betas=(0.9, 0.99),
    ),
    render=dict(
        type='Adam',
        lr=2e-4,
        eps=1e-15,
        betas=(0.9, 0.99),
    ),
    light=dict(
        type='Adam',
        lr=2e-4,
        eps=1e-15,
        betas=(0.9, 0.99),
    ),
    sdf=dict(
        type='Adam',
        lr=1e-7,
        eps=1e-15,
        betas=(0.9, 0.99),
    )

)


dataset = dict(
    type='BlenderDatasetStage2',
    # choose a scene: gummybear
    dataset_dir='./data/gummybear',
)

encoder = dict(
    nerf_pos_encoder=dict(
        type='FrequencyEncoder',
        multires=10,
        input_dims=4,
    ),
    nerf_dir_encoder=dict(
        type='FrequencyEncoder',
        multires=4,
        input_dims=3,
    ),
    sdf_encoder=dict(
        type='FrequencyEncoder',
        multires=6,
        input_dims=3,
    ),
    nodir_encoder=dict(
        type='FrequencyEncoder',
        multires=-1,
        input_dims=3,
    ),
    inner_encoder=dict(
        type='FrequencyEncoder',
        multires=4,
        input_dims=3,
    ),
)

ckpt = dict(
    stage1_ckpt_path="/home/user/JNeuralTO/JNeRF/log/gummybear/womask/checkpoints/ckpt_100000.pkl",
    model_ckpt_path=None,
    ckpt_file=None
)

loss = dict(
    eik_weight=0.01,
    roughrange_weight=0.01,
    smoothness_weight=0.01
)

lr = dict(
    material=2e-4,
    geometry=1e-7,
    render=2e-4,
    light=2e-4
)

model = dict(
    d_out=257,
    d_hidden=256,
    n_layers=8,
    skip_in=[4],
    bias=0.5,
    scale=1.0,
    geometric_init=True,
    weight_norm=True,
)

material = dict(
    type='NeuralTOMaterial',
    diffuse_albedo_network=dict(
        d_feature=256,
        mode='idr',
        d_in=9,
        d_out=3,
        d_hidden=256,
        n_layers=4,
        weight_norm=True,
        squeeze_out=True,
        network_type="nodir_encoder"
    ),
    trans_albedo_network=dict(
        d_feature=256,
        mode='no_view_dir',
        d_in=6,
        d_out=1,
        d_hidden=256,
        n_layers=4,
        squeeze_out=False,
        output_bias=0.1,
        output_scale=1.0,
        network_type="nodir_encoder"
    ),
    roughness_network=dict(
        d_feature=256,
        mode='no_view_dir',
        d_in=6,
        d_out=1,
        d_hidden=256,
        n_layers=4,
        squeeze_out=False,
        output_bias=0.1,
        output_scale=1.0,
        network_type="nodir_encoder"
    )
)

light = dict(
    init_light_intensity = 5.0
)

render = dict(
    add_ss="all",
    MipSSS=dict(
        mipmap_scale = 0.577,
        dirs_multires = 4,
        points_multires = 10,
        predict_sigma = True,
        predict_color = True
    )
)


base_exp_dir = './log/gummybear_s2/womask'
recording = ['./', './models']
save_freq = 10000
batch_size = 100
end_iter = 50001
report_freq = 1000
val_freq = 5000
log_freq = 5000

use_mask = False
background_color = None