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
```
Required package list
```bash
pandas>=2.1.4
numpy>=1.26.4
torch-geometric==2.7.0
scikit-learn>=1.2.2
scipy>=1.11.4
```
Data format

the drug dataset Data format must be .csv (comma-separated values).

the first column is drug name, and the second column must be unique SMILES strings.

For a detailed format, refer to Drug_SMILES.csv.

the PTM dataset Data format must be .csv (comma-separated values).

the first column is PTM name, and the second column must be unique PTM peptide window sequences.

For a detailed format, refer to Clean_data/PTM_peptide.csv.

the drug-PTM interaction Data format must be .csv (comma-separated values).

interaction Data, column = ['Drug', 'PTM'].

For a detailed format, refer to DrugPTM_rawdata.csv.

PBIFNet help
the PBIFNet.py in PBIFNet file is a python script (python==3.11.7) for convenience using PBIFNet by Command node
The introduction of the PBIFNet parameters can be achieved by naming them as follows:
```bash
cd PBIFNet
python PBIFNet.py -h
```
Output

```Bash
$ python PBIFNet.py -h
usage: PBIFNet.py [-h] [--K K] [-b BATCH_SIZE] [-e EPOCHS] [--lr LR]

PBIFNet: A Pre-trained Language Model and Induced-Fit Mechanism-Based
Framework for Drug-PTM Interaction Prediction.

options:
  -h, --help            show this help message and exit
  --K K                 Number of latent subspaces. Default is 2.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training and evaluation. Default is 128.
  -e EPOCHS, --epochs EPOCHS
                        Number of training epochs per fold. Default is 100.
  --lr LR               Learning rate. Default is 1e-4.
```

If there are any problems, please contact me.
Shiyun Liu, E-mail: 19898078438@163.com
