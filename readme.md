## Named entity recognition (NER)
[label](https://www.google.com/url?sa%3Di%26url%3Dhttps%3A%2F%2Fopen.spotify.com%2Falbum%2F2CrwABl5o2fAN73esT3VWs%26psig%3DAOvVaw0Ha6LgRDD7wtYUGI6N-FLO%26ust%3D1672152196760000%26source%3Dimages%26cd%3Dvfe%26ved%3D0CA0QjRxqFwoTCKC6s8HCl_wCFQAAAAAdAAAAABAD)

NER is a task in NLP that aims to extract entities in a text. An entity can be a person, city, country, etc., and can comprise of a single or multiple words.

This is a project on named entity recognition using HuggingFace's DistilBert model. The DistilBert model was proposed as a smaller, faster and lighter (i.e. 'distilled') version of BERT, preserving over 95% of BERT's performance while reducing the number of parameters by 40% and running faster by 60% (https://arxiv.org/abs/1910.01108)

## The approach
1. Tokenize text
2. Align labels with tokenized texts - this is a problem because a single word can be split into a few tokens, and therefore the number of tokens > number of labels. Fortunately DistilBert Tokenizer's word_id can be used to rectify this problem - set all subsqeuent token of the word (except the first) with an ID of -100. 
3. Load HuggingFace's pre-trained DistilBert model and train it over 1-5 epochs. There isn't a need for any extensive training because DistilBert is a pre-trained transformer.