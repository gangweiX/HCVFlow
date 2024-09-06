# HCVFlow
Hybrid Cost Volume for Memory-Efficient Optical Flow <br/>
Yang Zhao*, Gangwei Xu*, Gang Wu <br/>
\* denotes equal contribution.

## Network architecture
![image](figures/network.png)

## Environment
* NVIDIA RTX 3090
* python 3.8
* torch 1.12.1+cu113

### Create a virtual environment and activate it.

```Shell
conda create -n hcvflow python=3.8
conda activate hcvflow
```
### Dependencies

```Shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install tensorboardX
pip install opencv-python
pip install scipy
pip install scikit-image
pip install matplotlib
```
