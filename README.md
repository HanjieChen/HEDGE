# HEDGE
Code for the paper ["Generating Hierarchical Explanations on Text Classification via Feature Interaction Detection"](https://arxiv.org/abs/2004.02015)

### Requirement:
- torchtext == 0.4.0
- gensim == 3.4.0
- pytorch == 1.2.0
- numpy == 1.16.4

We provide the example code of HEDGE interpreting the LSTM, CNN and BERT model on the IMDB dataset. We adopt the BERT-base model built by huggingface: https://github.com/huggingface/transformers.

In each folder, run the following command:
```
python hedge_main_model_imdb.py --save /path/to/your/model
```

To visualize the hierarchical explanation of a sentence, run
```
python hedge_main_model_imdb.py --save /path/to/your/model --visualize 1(the index of the sentence)
```

### Reference:
If you find this repository helpful, please cite our paper:
```bibtex
@article{chen2020generating,
  title={Generating hierarchical explanations on text classification via feature interaction detection},
  author={Chen, Hanjie and Zheng, Guangtao and Ji, Yangfeng},
  journal={arXiv preprint arXiv:2004.02015},
  year={2020}
}
```
