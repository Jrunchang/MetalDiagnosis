conda create -n MetalDiagnosis python==3.9 -y
source activate MetalDiagnosis
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric==2.2.0
pip install scikit-learn
pip install networkx
pip install griphein==1.7.7
pip install e3nn==0.5.1
pip install biopython==1.81


