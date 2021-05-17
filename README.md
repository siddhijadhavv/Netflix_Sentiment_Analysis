# Netflix_Sentiment_Analysis

This is a Machine Learning project on the very famous Netflix sentiment analysis using NLP. Based on the reviews given by the customers, we can determine whether the show or the movie was good or bad.

Following is the folder structure :
1. /input
  - Consists of the two datasets seperated by the means of positive or negative reviews.
2. /notebooks
  - Consists of the ipynb file of the project where the project is implemented using SVM and Naive Bayes model.
3. /src
  - Consits of the various python files where the code is written.
4. /models
  - Consists of the models saved.
5. /vectorizers
  - Consists of the vecorized files saved for each model.
  
## Prerequisites
- For referencing the notebook, you will need Jupyter Notebooks installed.
- Python version on which the model was built - Python 3.8.3
- Any Python IDE ( I have used VS Code )

## How to run the file?
- Open the file in any IDE and open the terminal.
- Use the below command :
    python train.py --model <model_name>
  NOTE: In place of <model_name>, mention one of the models from the below (as mentioned in the model_dispatcher.py)
    1. decision_tree_gini ( Decision tree classifier using gini )
    2. decision_tree_entropy ( Decision tree classifier using entropy )
    3. rf ( Random forest classifier )
    4. naive_bayes_gaussian ( Naive Bayes using Gaussian distribution )
    5. naive_bayes_multinominal( Naive Bayes using Multinominal distribution )
