## Preprocess the dataset
To prepare the dataset for model training, follow these streamlined steps in our data preprocessing pipeline. 

  - Initiate the process by running the dataset through the `EDA_Stage1.ipynb`, which encompasses exploratory data analysis techniques
    tailored to extract meaningful insights and handle initial data transformations.

  - Subsequently, progress to the `EDA_Stage2.ipynb`, where along with the original data, further refinement and preprocessing
    are carried out to shape the data into its optimal form for model compatibility. This sequential application of exploratory
    data analysis ensures that the dataset is meticulously curated and ready for seamless integration into our model training process.

## Likes prediction
The data is then loaded into the BM25 model, which, based on a certain set of parameters, decides which XGBoost model to be called. 
We employ the binning technique to decide the XGBoost model to be called. According to the decision, the respective model is invoked and prediction is generated.
