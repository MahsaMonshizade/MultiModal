# MultiModal

* To view a live preview of the compiled Markdown Pressing Cmd + Shift + V (on Mac) 

### Conda env

My conda version: 24.5.0

```
conda create --name MultiModal
conda activate MultiModal
conda install pandas
conda install openpyxl
conda install matplotlib
conda install numpy
conda install anaconda::scikit-learn
conda install anaconda::scipy
conda install conda-forge::tqdm
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install anaconda::seaborn
```

metagenomics --> samples: 2891 , features: 578

metatranscriptomics --> samples: 2891 , features: 421  

metabolomics --> samples: 2891 , features: 596

Always a good practice to define a random input output tp see if problems are from your data or your model or the way you run your model
for dataset it's a good practice to normalize data 
also some problems in evaluation part