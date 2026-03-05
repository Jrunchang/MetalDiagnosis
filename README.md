# MetalDiagnosis
MetalDiagnosis combines an improved equivariant graph neural network and a protein pre-trained language model to predict disease-related mutation sites in metal binding proteins.
<img width="9638" height="9538" alt="workflow" src="https://github.com/user-attachments/assets/30ecc24f-cdf5-41cc-aeb5-7e06bfa11d46" />
# Step 1: Clone the GitHub repository
```bash
git clone https://github.com/MetalDiagnosis
cd MetalDiagnosis
```
# Step 2: Build required dependencies
It is recommended to use Anaconda to install PyTorch, PyTorch Geometrics and other required Python libraries.
'''bash
source install.sh
'''

# Step 3: Download required software
First, Using API to predict ESMC embedding online at https://github.com/evolutionaryscale/esm#esm-c-forge-

Then constructing protein graph and extracting node features from graphein(https://github.com/a-r-j/graphein)

# Step 4: Running MetalDiagnosis
'''bash
python test.py -i test_dataset.pt 
'''

# Other
We provide prediction results for an independent test set and 611 sites with **uncertain significance**.
