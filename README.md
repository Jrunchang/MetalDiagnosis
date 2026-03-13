# MetalDiagnosis
MetalDiagnosis combines an improved equivariant graph neural network and protein pre-trained language model ESMC to predict disease-related mutation sites in metal binding proteins.
<img width="9638" height="9538" alt="workflow" src="https://github.com/user-attachments/assets/21ac0d94-351e-415b-bb6d-27d44dcdfaa0" />

# Step 1: Clone the GitHub repository
```bash
git clone https://github.com/MetalDiagnosis
cd MetalDiagnosis
```
# Step 2: Build required dependencies
It is recommended to use Anaconda to install PyTorch, PyTorch Geometrics and other required Python libraries.
```bash
source install.sh
```

# Step 3: Download required software
First, Using API to extract ESMC embeddings online at (https://github.com/evolutionaryscale/esm#esm-c-forge-) using ESMC_embedding_extract.ipynb.

Then constructing protein graph and extracting node features from graphein(https://github.com/a-r-j/graphein) using construct_graph&node_feature.ipynb.

# Step 4: Running MetalDiagnosis on the independent test set.
```bash
python test.py -i test_dataset.pt 
```

# Other
We provide prediction results for an independent test set and 611 sites with **uncertain significance**.
