# SMCS_FraudDetection

---
## Environment Setup

***Be sure to first create and activate your Python environment! (via Conda or Python's virtual environments)***

Due to Pytorch's version conflicts with other packages, we recommend that you replicate our environment using `setup_env.sh`, which can be used in two ways, depending on the desired hardware:

### CPU 

```
bash setup_env.sh -d cpu
```

### GPU
```
bash setup_env.sh -d gpu -c cu124
```

`-c` is the CUDA version (when using NVIDIA GPUs) currently installed and used by your GPU's drivers. Use the version you see when running `nvidia-smi`.

---

## Training a model
```
bash model_training.sh
```

Inside the script you can change the training environment, data, model, etc.
