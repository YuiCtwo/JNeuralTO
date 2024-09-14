# NeuralTO

Implementation of paper *NeuralTO: Neural Reconstruction and View Synthesis of Translucent Objects*(TOG' 24) using [Jittor](https://github.com/Jittor/JNeRF) framework.

# Environment Setup

Please check the official JNeRF environment requirements before running our project and follow the installation command to setup a Jittor environment.

```bash
conda create -n JNeuralTO python=3.7
conda activate JNeuralTO
cd /path/of/our/project
cd JNeRF
python -m pip install --user -e .
```

Installing cupy for JNeuS
```bash
# for CUDA-11.3, run: 
pip install cupy-cuda113
```

# Configuration Setup


# Running

- For reconstruction in Stage1, run the command below.
It takes about 5h in one NVIDIA 3090 GPU using the example config file.

```shell
python tools/run_net.py \
       --config-file ./projects/neuralto/configs/syn_gummybear_womask.py \
       --type neuralTO_recon \
       --task train
```

- You can extract mesh using this command. Make sure that you have set a proper bounding box for the model. By default, we use 
`object_bbox_min=[-1.21, -1.21, -1.21]` and `object_bbox_max=[1.21, 1.21, 1.21]`
```shell
python tools/run_net.py \
       --config-file ./projects/neuralto/configs/syn_gummybear_womask.py \
       --type neuralTO_recon \
       --task validate_mesh
```

- For learning scattering property in Stage2, we rely on the reconstructed geometry.
If all things are done, run the command below:

```shell
python tools/run_net.py \
       --config-file ./projects/neuralto/configs/syn_gummybear_sss.py \
       --type neuralTO_render \
       --task train
```