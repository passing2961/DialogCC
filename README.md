<h1 align="center"> Welcome to DialogCC </h1>

This is the official repository for our NAACL 2024 paper: DialogCC: An Automated Pipeline for Creating High-Quality Multi-modal Dialogue Datasets

As sharing images in an instant message is a crucial factor, there has been active research on learning an image-text multi-modal dialogue models.
However, training a well-generalized multi-modal dialogue model remains challenging due to the low quality and limited diversity of images per dialogue in existing multi-modal dialogue datasets.
In this paper, we propose an automated pipeline to construct a multi-modal dialogue dataset, ensuring both dialogue quality and image diversity without requiring minimum human effort. 
In our pipeline, to guarantee the coherence between images and dialogue, we prompt GPT-4 to infer potential image-sharing moments - specifically, the utterance, speaker, rationale, and image description. 
Furthermore, we leverage CLIP similarity to maintain consistency between aligned multiple images to the utterance.
Through this pipeline, we introduce DialogCC, a high-quality and diverse multi-modal dialogue dataset that surpasses existing datasets in terms of quality and diversity in human evaluation.
Our comprehensive experiments highlight that when multi-modal dialogue models are trained using our dataset, their generalization performance on unseen dialogue datasets is significantly enhanced. We will release the source code and dataset following publication.

## Pipeline

## Dataset

## To-Do List

## Citation