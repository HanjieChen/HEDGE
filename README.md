# HEDGE
Code for the paper ["Generating Hierarchical Explanations on Text Classification via Feature Interaction Detection"](https://arxiv.org/abs/2004.02015)

Requirement:
- torchtext == 0.4.0
- gensim == 3.4.0
- pytorch == 1.2.0
- numpy == 1.16.4

We provide the example code of HEDGE interpreting the LSTM, CNN and BERT model on the IMDB dataset. We adopt the BERT-base model built by huggingface: https://github.com/huggingface/transformers.

To run the code, use the following command:
```
python hedge_main.py --save /path/to/your/model --out-file hedge.txt
```

To visualize the hierarchical interpretation of a sentence, uncomment line 91 and line 120, and comment line 90. You can set the index of the sentence that you want to visualize.




