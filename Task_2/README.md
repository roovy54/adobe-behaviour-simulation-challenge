## Criteria Selection
This implementation offers an extensive approach to gathering tweet data, employing a multifaceted criteria selection. 
The retrieval process is finely tuned to capture tweets based on likes, company affiliation, date parameters, and visual 
content such as images and videos. The initial three criteria—likes, company, and date—are harnessed through a straightforward 
yet effective L1 algorithm, ensuring precision in data extraction.

## Contextual Similarity
For contextual visual similarity, a sophisticated approach is adopted, leveraging language-based embeddings. Here, we make use of
the embeddings that were generated through the languagebind. This method goes beyond conventional methods, providing nuanced 
representations that capture the semantic essence of visual content associated with tweets.

## Analogical Prompting
A distinctive feature of this implementation lies in its utilization of a `analogical` prompting technique. 
This technique, coupled with the wealth of tweet data collected, serves as a catalyst for generating new tweets. 
The generation process is powered by LLaVa-7b, a state-of-the-art language model. This integrated methodology not only
enriches the diversity and relevance of the generated content but also showcases the innovative synergy between data retrieval, 
language embeddings, and advanced language models in the realm of tweet generation.
