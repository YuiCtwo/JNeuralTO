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

You can testing your enviroment using the command below:
```bash
python tools/test_env.py
```

# Dataset Setup


# Running

```
python tools/run_net.py --config-file ./projects/neuralto/configs/syn_gummybear_womask.py --type neuralTO_recon --task train
```