# HEDGE
Code for the paper ["Generating Hierarchical Explanations on Text Classification via Feature Interaction Detection"](https://arxiv.org/abs/2004.02015)

Requirement:
- torchtext == 0.4.0
- gensim == 3.4.0
- pytorch == 1.2.0
- numpy == 1.16.4

The example code use an LSTM model trained on the IMDB dataset. Your can change to your dataset by modifying `load_data.py`. To run the code, use the following command:
```
python hedge_main.py --save /path/to/your/model --out-file hedge.txt
```

To visualize the hierarchical interpretation of one sentence, uncomment line 91 and line 120, and comment line 90. Then in line 91, set the index of the sentence that you want to visualize.




