<h1 align="center"> Welcome to DialogCC </h1>

This is the official repository for our NAACL 2024 paper: DialogCC: An Automated Pipeline for Creating High-Quality Multi-modal Dialogue Datasets

As sharing images in an instant message is a crucial factor, there has been active research on learning an image-text multi-modal dialogue models.
However, training a well-generalized multi-modal dialogue model remains challenging due to the low quality and limited diversity of images per dialogue in existing multi-modal dialogue datasets.
In this paper, we propose an automated pipeline to construct a multi-modal dialogue dataset, ensuring both dialogue quality and image diversity without requiring minimum human effort. 
In our pipeline, to guarantee the coherence between images and dialogue, we prompt GPT-4 to infer potential image-sharing moments - specifically, the utterance, speaker, rationale, and image description. 
Furthermore, we leverage CLIP similarity to maintain consistency between aligned multiple images to the utterance.
Through this pipeline, we introduce DialogCC, a high-quality and diverse multi-modal dialogue dataset that surpasses existing datasets in terms of quality and diversity in human evaluation.
Our comprehensive experiments highlight that when multi-modal dialogue models are trained using our dataset, their generalization performance on unseen dialogue datasets is significantly enhanced. We will release the source code and dataset following publication.

## How to run Pipeline?

Currently, this repository only supports generating appropriate image-sharing moments using GPT-4. We will release the complete code of our proposed pipeline: (1) Collecting, (2) Aligning, and (3) Filtering.

```
python run.py \
  --run-id test \
  --model gpt-4 \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-tokens 1024 \
  --frequency-penalty 0.0 \
  --presence-penalty 0.0
```

## DialogCC

You can now load DialogCC from the [HuggingFace hub](https://huggingface.co/datasets/passing2961/dialogcc) as the following:
```python
from datasets import load_dataset

dataset = load_dataset("passing2961/dialogcc")
```

> 🚨 Disclaimer: Despite our efforts to create a high-quality and diverse multi-modal dialogue dataset, it still contains harmful content, such as social bias. Moreover, since DialogCC incorporates dialogues from the DailyDialog dataset, which is licensed under CC BY-NC-SA 4.0, DialogCC is shared under the license CC-BY-NC-SA 4.0. Therefore, we strongly recommend using our dataset for academic and research purposes.

## Citation

```
@inproceedings{lee2024dialogcc,
  title={DialogCC: An Automated Pipeline for Creating High-Quality Multi-Modal Dialogue Dataset},
  author={Lee, Young-Jun and Ko, Byungsoo and Kim, Han-Gyu and Hyeon, Jonghwan and Choi, Ho-Jin},
  booktitle={Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={1938--1963},
  year={2024}
}
```
