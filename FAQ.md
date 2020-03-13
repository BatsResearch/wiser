# WISER FAQ
## Pipeline Questions

- What is the best practice to writing tagging and linking rules?
    - It depends! It is generally best to write tagging rules of high accuracy and coverage.
       We generally recommend writing tagging rules that identify one or two specific keyword tokens in entity spans, 
       and using the linking rules to define the boundaries of said entities.  
 
- How many tagging rules should I write?
    - The number of tagging rules depends on the task you're working on. 
    Typically, rules with high accuracy won't hurt performance, so you can write as many of these as you'd like.
    The best way to evaluate your tagging rules is using the majority vote.
    To develop robust NER frameworks, we recommend introducing a few tagging rules that provide negative supervision 
    in the form of 'O' tags.

- How many linking rules should I write?
  - Generally speaking, the better the linking rules, the better the performance of the linked HMM model! 
  You should aim to write many linking rules that have high coverage and accuracy (+90%). 

- Is it better to write rules with high coverage or high accuracy?
  - High accuracy is generally associated with high precision, whereas high coverage is associated with high recall.
  While you should aim to strike a balance between the two types of rules, we recommend prioritizing precision over recall.
  This is because the discriminative neural network improves the generalization of the pipeline, 
  and generally provides increases in recall. 

## Debugging

- The majority vote scores are larger than the generative model scores. 
    - If your generative model is underperforming the majority vote, that means the model is weighting
    the tagging and linking rules worse than an unweighted majority vote. 
    To fix this issue, try reducing the strength of the regularization of your generative model 
    (``acc prior`` and ``balance prior``).
    Doing so will increase the confidence in the votes outputted by the tagging and linking rules.
    
- The discriminative neural network scores are significantly worse than the generative model scores.
    - If you observe that your discriminative model is significantly underperforming your generative mode,
    then you should try the following:
        - Tuning the generative hyperparameters: 
        Oftentimes, the probabilistic labels produced by the generative model are very close to uniform. 
        In such cases, the discriminative model can sometimes fail to learn from the underlying distribution. 
        Reducing the values of ``acc prior`` and/or ``balance prior`` will generally lead to increased performance 
        (see paper for more details about generative hyperparameters).
        - Tuning the discriminative hyperparameters: 
        The discriminative performance can also be dependent on the complexity of the neural tagger. 
        Try adjusting the number of hidden layers, the hidden size, or the batch size.
        - Running a grid search using different generative hyperparameter values, and training the discriminative 
        neural tagger with the best-performing values. 
        
- The discriminative dev F1 looks good, but the test F1 is significantly worse.
    - Low test F1 is generally associated to dev data overfitting. 
    We recommend going over the tagging rules and making sure that they're not overfitting to the train or dev data.
    You should also revise the discriminative hyperparameters,
    and reduce or increase the complexity of the model accordingly.
  
- My generative pipeline is looking good, but I get los scores on the train sets while training the discrimnative model.
    - Since our framework does not require training labels, the neural tagger there will have no ground-truth labels to 
    compute training precision, recall, and F1. during the training stages, train precision, recall, and F1 will 
    not be displayed erroneously and be very close to 0.

 