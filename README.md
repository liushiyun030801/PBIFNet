# PBIFNet

PBIFNet: A Prompt-Guided Bi-directional Induced Fit Network for Drug-PTM Interaction Prediction

## Install dependencies

*   Clone the repository.

```bash
git clone https://github.com/liushiyun030801/PBIFNet.git
```
*  The package is developed based on the Python libraries torch and torch-geometric (PyTorch Geometric) framework, and can be run on GPU (recommend) or CPU.
(i) torch (CPU)

```bash
pip3 install torch torchvision torchaudio
```
(ii) torch (GPU)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
Install other the required packages.
```bash
pip install -r requirement.txt
Required package list
pandas>=2.2.2
numpy>=1.26.4
torch-geometric==2.6.1
gseapy>=1.1.9
networkx>=3.2.1
iterative-stratification>=0.1.9
scikit-learn>=1.4.2
```
