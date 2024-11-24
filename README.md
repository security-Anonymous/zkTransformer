# zkGPT

## Introduction

This is the implementation of zkGPT, which is a SNARK for LLM inference. 
This project partially references code from the work of zkCNN and Lasso.
**Currently this version only supports GPT-2.** 



## Requirement

- C++14
- cmake >= 3.10
- GMP library


## Experiment Script
### Clone the repo
To run the code, make sure you clone with
``` bash
git clone --recurse-submodules git@github.com:security-Anonymous/zkTransformer.git
```
since the polynomial commitment is included as a submodule.

### Install GMP Library
```
sudo apt install libgmp-dev
```

### Run a demo of proving LLM inference
The script to run LLM inference proving.
``` bash
./llm.sh
```


## Reference
- [zkCNN: Zero knowledge proofs for convolutional neural network predictions and accuracy](https://doi.org/10.1145/3460120.3485379).

- [Unlocking the lookup singularity with Lasso](https://eprint.iacr.org/2023/1216)

- [mcl](https://github.com/herumi/mcl)
