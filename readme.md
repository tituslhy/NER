This is a project on named entity recognition using HuggingFace's DistilBert model. The DistilBert model was proposed as a smaller, faster and lighter (i.e. 'distilled') version of BERT, preserving over 95% of BERT's performance while reducing the number of parameters by 40% and running faster by 60% (https://arxiv.org/abs/1910.01108)

## The approach
1. Tokenize text
2. Align labels with tokenized texts - this is a problem because a single word can be split into a few tokens, and therefore the number of tokens > number of labels. Fortunately DistilBert Tokenizer's word_id can be used to rectify this problem - set all subsqeuent token of the word (except the first) with an ID of -100. 
3. Load HuggingFace's pre-trained DistilBert model and train it over 1-5 epochs. There isn't a need for any extensive training because DistilBert is a pre-trained transformer.