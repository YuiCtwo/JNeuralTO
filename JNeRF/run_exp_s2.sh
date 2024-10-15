export JT_SAVE_MEM=1
export cpu_mem_limit=16000000000
python tools/run_net.py \
        --config-file ./projects/neuralto/configs/syn_gummybear_sss.py \
        --type neuralTO_render \
        --task train