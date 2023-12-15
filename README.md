# Team 31 - Adobe Behaviour and Content Simulation Challenge

Communication in the digital realm is a dynamic interplay between senders and receivers, with each message crafted to evoke specific user behaviors. For marketers, the ultimate goal is to understand and predict user engagement – the likes, comments, shares, and purchases that define the success of their content.

In today's fast-paced digital landscape, businesses face the ever-growing challenge of delivering exceptional customer experiences. Adobe Experience Cloud, a comprehensive suite of tools, stands at the forefront, empowering businesses to design and deliver outstanding customer journeys. A seamless and engaging customer journey not only drives user satisfaction but also plays a pivotal role in brand differentiation, market credibility, and overall business success.

This challenge presents two critical aspects of customer engagement:

### Tweet Likes Prediction

Predicting tweet likes accurately is essential for understanding user engagement. Given the content of a tweet (text, company, username, media URLs, timestamp), predict its user engagement, specifically the number of likes.

### Tweet Content Generation

Predicting tweet likes accurately is essential for understanding user engagement. Given tweet metadata (company, username, media URL, timestamp), generate the tweet text. This task delves into the art and science of crafting compelling content that resonates with the audience.

## Repository Structure

- **Data_Preprocessing**: 

    This folder contains all the necessary files and sub-folders related to data preprocessing.

    - **EDA (Exploratory Data Analysis)**: This section is divided into two Jupyter notebooks:

        - **EDA_Stage1.ipynb**: The first stage of exploratory data analysis.
        - **EDA_Stage2.ipynb**: The second stage of exploratory data analysis.

    - **Embeddings**: Consists of various embedding files and notebooks:

        - **languagebind**: A folder containing resources and files related to language binding.
        - **Audio_embeds.ipynb**: A Jupyter notebook for audio embeddings.
        - **EVA_CLIP.ipynb**:  A Jupyter notebook associated with EVA and CLIP embeddings.
        - **Jina_Embeds.ipynb**: Generating content text embeddings using the Jina.
        - **Merge_embed.ipynb**: Concatenating the embedding vectors.

    -  **Experiments**:

        This folder contains the experimental models and logs.

        -  **keras_model.ipynb**: A notebook containing experiments with Keras models.
        -  **prompt_log_0-100.txt**: Log file for prompts having 0 to 100 likes.
            -  **prompt_log_100-300.txt**: Log file for prompts having 100 to 3-5K likes.

    -  **Preprocessing**:

        This folder contains the experimental models and logs.

        -  **Image OCR**: A notebook containing experiments with Keras models.
        -  **Video OCR**: Log file for prompts having 0 to 100 likes.
        -  **Image and Video Captioning**: A notebook containing the code for generating image and video captions.

-  **Task 1**:

    This folder contains the models employed for Task 1.

    -  **Analogical Retrievers**
    -  **Experiments**: Cross Attention, jina embed, merge embeds, ResNet+BERT, RoBERTa, ViT+RoBERTa implementations.
    -  **XG Boost Ensemble**: Implemented XG Boost classifier, regressor ensembled with another regressor layer to create final meta model.

-  **Task 2**:

    This folder contains the models employed for Task 2.

    -  **Analogical Retrievers**: Folder containing resources for analogical retrievers.
    -  **Experiments**: Folder with additional experimental files and data.
    -  **Encoder.py**: Python script for encoder implementation.

The repository also includes a README.md file that provides an overview and explanation of the content within this repository.
