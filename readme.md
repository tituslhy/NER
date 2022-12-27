## Named entity recognition (NER)
<p align="center">
  <img src="Images/jamesbond.jpeg">
</p>

NER is a task in NLP that aims to extract entities in a text. An entity can be a person, city, country, etc., and can comprise of a single ("James") or multiple words ("James Bond").

This is a NER project using HuggingFace's pre-trained DistilBert model. The DistilBert model was proposed as a smaller, faster and lighter (i.e. 'distilled') version of BERT, preserving over 95% of BERT's performance while reducing the number of parameters by 40% and running faster by 60% (https://arxiv.org/abs/1910.01108)

This project is a personal endeavor to learn more about the domain of NLP by doing, and closely follows the code and elaboration of the Medium article https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a 

## The approach
1. Tokenize text
2. Align labels with tokenized texts - this is a problem because a single word can be split into a few tokens, and therefore the number of tokens > number of labels. Fortunately DistilBert Tokenizer's word_id can be used to rectify this problem - set all subsqeuent token of the word (except the first) with an ID of -100. 
3. Load HuggingFace's pre-trained DistilBert model and train it over 1-5 epochs. There isn't a need for any extensive training because DistilBert is a pre-trained transformer.
<br>

## To instantiate environment
```
git clone https://github.com/tituslhy/NER
pip -r requirements.txt
```
<br>

## Training the model
The flags of the python3 run are:
1. --batch: batch size. Defaults to 2
2. --epochs: Number of training epochs. Defaults to 3.
3. --lr: Learning rate. Defaults to 1e-3.
4. --path: Path for data. Defaults to ner.csv
```
python3 train.py --batch 4 --epochs 5 --lr 0.01 --path data.csv
```

## Using the model to run inferences
The flags of the python3 run are:
1. --txts: This is the sentence of interest and is a required argument
2. --pretrained: Boolean. Defaults to False. If True, the script looks within the directory for "best_model.pt". Otherwise it uses the pretrained DistilBert model to run inference.
```
python3 inference.py --txts "Bill Gates is the founder of Microsoft" --pretrained True
```