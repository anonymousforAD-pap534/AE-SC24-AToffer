# AToffer
The code and example data for paper "*AToffer*: Automatic Operator Fusion Framework for Sparse Transformer Inference"

## Getting Started
+ Hardware 
    + NVIDIA GPU RTX3090 and RTX4090
+ OS & Software
    + ubuntu 20.04
    + GCC == 11.4.0
    + CUDA == 12.2
    + numpy >= 1.23.5
    + torch >= 2.1.1
    + nijia >= 1.11.1

To set up the dependencies, run the following command:
```shell
pip install -r requirements.txt
```

## File Organization
+ `script/`: Python script of comparison experiments. hash encoder and search mechanism in *AToffer* 
    + `syncfree.py`: the basic script using customized kernels to verify whether the environment is configured correctly.
    + `eva_A_MHA_script.py`: the script to run comparsion evaluation of MHA 
    + `eva_B_end2end_script/`: run comparsion evaluation of End2end forward of BERT forward.
    + `eva_C_search_script/`: 5round * 128 iter to search the optimal fusion scheme in search space.
    + `eva_D_cost_analysis_script/`: the script to analyze the synchronization cost and search cost.
    + `forward.py`: a complete forward calculation process for BERT.
    + `generated_genidx.py`: generate code with a specific fusion scheme, containing Hash encoder.  
+ `src/`: CUDA code of customized kernels for MHA in sparse Transformer.
+ `utils/`: some components that are frequently used in code.
+ `example_data/`: the raw experiments result on our device.
+ `plt_script/`: the python script to draw the performance figure.


## Reproduce implement in *AToffer*
1. To verify the environment, enter the directory of `script` and try to execute a demo script using customized kernels.
```shell
# batch_size = 8, layer_num = 12, seq_len = 256, head_num = 12, head_size = 64
# mask_id = 1 (type of sparse mask: 1 for strided, 2 for fixed)
python syncfree.py 8 1 256 12 64 1
```
2. Run script to get performance result of MHA, we warm up 10 times and run it 100 times to get an average value.
```shell
# platform = Nvidia_3090, this according to your platform
python eva_A_MHA_script.py Nvidia_3090
```
3. Comparsion evaluation with Native PyTorch & PyTorch JIT of End2end forward of BERT forward.
```shell
cd script/eva_B_end2end_script
# platform = Nvidia_3090, this according to your platform
python B_batchtest.py Nvidia_3090
```

4. Run search script to get the optimized fusion scheme of sparse Transformer (take BERT forward for a study case)
```shell
cd script/eva_C_search_script
python C_search_all_script.py
```
5. Use [NVIDIA Nsight System](https://developer.nvidia.com/nsight-systems) to analyze the synchronization cost whose ratio is displayed by *cudaStreamSynchronize* 
```shell
# seq_len = 64, 128, 256, 384, 512, 768, 1024
cd script/eva_D_cost_analysis_script
nsys profile --stat=true -o base_rep64 python eva_D1_SyncCost_base.py 64
nsys profile --stat=true -o ours_rep64 python eva_D1_SyncCost_ours.py 64
```

6. To analyze the overhead of reward-based search mechanism in *AToffer*
```shell
# seq_len = 256 mask_id = 1 (type of sparse mask: 1 for strided, 2 for fixed)
cd script/eva_D_cost_analysis_script
python D2_search_cost.py 256 1
```