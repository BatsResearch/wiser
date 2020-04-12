# WISER FAQ
## Pipeline Questions

-  Hows does the IOB1 tagging scheme work, and how to I generate tagging votes?
    - The IOB1 format (short for inside, outside, beginning 1) is one of the most common tagging formats for many NER 
    tasks.   
        - An I-prefix  tag indicates the beginning or inner tag of a chunk that is not preceded by another chunk. 
        Alongside O tags, these are the only tags that we recommend modeling in your pipelines.
        - An B-prefix tag indicates the beginning tag of a chunk that is preceded by another chunk. Given the relatively 
        low frequency of B-tags, we rarely need to write tagging for B-tags.
        - An O-tag indicates that the corresponding token is not an entity of interest. These tags are commonly assigned 
        to punctuation signs, and entities that are neither I-tags nor B-tags. You should always aim to have a few 
        negative supervision tagging rules that output O-tags.
    - The difference between IOB1 and other similar tagging schemes (e.g., IOB2, IOE1) can be found in the following
    [link](https://www.researchgate.net/figure/3-IOB1-IOB2-IOE1-and-IOE2-Base-Phrase-annotation-schemes_tbl7_50839830).     
    
- What is the best practice for writing tagging and linking rules?
    - It is generally best to write tagging rules of high accuracy and coverage. However, your specific rules should 
    depend on your task! If you are focusing identifying a few quality entities, you will want to prioritize accuracy 
    over coverage. However, if your goal is to extract a large number of entities from a corpus, you may be better off
    sacrificing accuracy for increased coverage. 
     
    - You should always produce an interplay between tagging and linking rules: we generally recommend writing tagging 
    rules that identify one or two specific keyword tokens in longer entity spans, and using the linking rules to define 
    the boundaries of said entities (e.g, writing a tagging rule that identifies surnames, and a linking rule that links
    the entire capitalized name).
 
- How many tagging rules should I write?
    - The number of tagging rules depends on the task you're working on. 
    Typically, rules with high accuracy (+90%) won't hurt performance, so you can write as many of these as you'd like.
    The best way to evaluate your tagging rules is using the majority vote (see intro. tutorial, notebook 2).
    Further, to develop robust NER frameworks, we recommend introducing a few tagging rules that provide negative 
    supervision in the form of 'O' tags.

- How many linking rules should I write?
    - Generally speaking, the better the linking rules, the better the performance of the linked HMM model! 
    You should aim to write many linking rules that have high coverage and accuracy (+90%). 

- Is it better to write rules with high coverage or high accuracy?
    - High accuracy is generally associated with high precision, whereas high coverage is associated with high recall.
    While you should aim to strike a balance between the two types of rules, for most tasks we recommend prioritizing 
    precision over recall. 
    This is because the discriminative neural network generalizes better when trained on quality, high-precision rules.
    In most cases, a large generative precision leads to significant increases in discriminative recall.


## Debugging

- The majority vote scores are larger than the generative model scores. 
    - If your generative model is underperforming the majority vote evaluation, that means your model is weighting
    the tagging and linking rules worse than an unweighted majority vote. 
    To fix this issue, try reducing the strength of the regularization of your generative model 
    (``acc_prior`` and ``balance_prior``).
    This will increase the confidence in the votes outputted by the tagging and linking rules.
    
- The discriminative neural network scores are significantly worse than the generative model scores.
    - If you observe that your discriminative model is significantly underperforming your generative model,
    then you should try the following:
        - Tuning the generative hyperparameters: 
        Oftentimes, the probabilistic labels produced by the generative model are very close to uniform. 
        In such cases, the discriminative model can sometimes fail to learn from the underlying distribution. 
        Reducing the values of ``acc_prior`` and/or ``balance_prior`` will generally lead to increased performance 
        (see paper for more details about generative hyperparameters).
        - Tuning the discriminative hyperparameters: 
        The discriminative performance can also be dependent on the complexity of the neural tagger. 
        Try adjusting the number of hidden layers, the hidden size, or the batch size (or all of them!).
        - Running a grid search using different generative hyperparameter values, and then training the discriminative 
        neural tagger with the best-performing set of values (see next item). 
        
- How do I run a grid search search over my generative model hyperparameters?
    - You will first need to create a small script that trains your generative model with different hyperparameters. We           
    recommend fixing the ``init_acc`` to 0.9 or 0.95, and then finding the optimal combination of ``acc_prior`` and               
    ``balance_prior`` over the values {0.5, 1, 5, 10, 50, 100, 500}. 
    However, sometimes the best combination of hyperparameters can produce a generative model whose probabilistic labels 
    are too close to uniform (e.g., good generative performance but poor discriminative performance). 
    In such cases, you may want to manually reduce the hyperparameters a bit (see previous item).
    
- The discriminative dev F1 looks good, but the test F1 is significantly worse.
    - Low test F1 is generally associated to overfitting to the dev data. 
    We recommend going over the tagging rules and making sure that they're not overfitting to the train or dev data 
    (e.g., writing rules that are too specific for some cases).
    You should also revise the discriminative hyperparameters, and change the model complexity of the accordingly.
  
- My generative pipeline is looking good, but I get poor scores on the training set while training the discrimnative 
model.
    - Since WISER does not require training labels, the neural tagger will have no ground-truth labels to 
    compute training scores. Therefore, during training, precision, recall, and F1 scores will be inaccurate and thus 
    close to 0.

## Suggestions and Contributing

- If you have any ideas or suggestions to improve WISER, please don't hesitate to post an issue! We're more than happy 
  to receive feedback and contributions from users.
