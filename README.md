# Urban Cleanliness Classification and Image Enhancement using AI

## Project Overview

This project focuses on using artificial intelligence to address urban cleanliness challenges. The goal is to classify urban scenes as clean or dirty and apply AI-based image enhancement techniques to visualize the potential transformation of dirty urban environments into clean ones.

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Models](#models)
  - [Task 1: Clean/Dirty Classification](#task-1-clean-dirty-classification)
  - [Task 2: Network Interpretation](#task-2-network-interpretation)
  - [Task 3: Clean/Dirty Image Transformation](#task-3-clean-dirty-image-transformation)
- [Conclusion](#conclusion)

## Introduction

Urban cleanliness is a growing challenge as cities expand rapidly. Excessive trash in urban areas is not only unsightly but also poses significant health hazards. This project explores the use of AI to classify urban scenes and propose visual enhancements for cleaner cities.

## Objectives

The primary objectives of this project are:

1. **Classify Urban Scenes**: Develop an AI model that can accurately distinguish between clean and dirty urban environments.
2. **Interpret AI Decisions**: Utilize Class Activation Maps (CAM) to understand the decision-making process of the AI model.
3. **Image Enhancement**: Apply CycleGAN to transform dirty urban scenes into clean ones, demonstrating the potential of AI in city maintenance.

## Dataset

This project uses the **Clean and Dirty Containers in Montevideo** dataset from Kaggle. The dataset consists of images categorized into clean and dirty urban environments, providing a foundation for training our models.

- **Number of Images**: 1206 clean and 1011 dirty for training, and 600 clean and 595 dirty for testing.
- **Image Types**: Mainly urban scenes with trash cans, standardized for identification and manipulation purposes.

## Models

### Task 1: Clean/Dirty Classification

We implemented a Convolutional Neural Network (CNN) using the **ResNet-18** architecture to classify urban scenes as clean or dirty.

- **Model Training**: Trained on the Clean and Dirty Containers dataset.
- **Evaluation Metrics**: Achieved a classification accuracy of **93.89%**.
- **Failure Case Analysis**: Misclassifications were primarily caused by factors such as lighting, occlusions, or similarities in texture between clean and dirty areas.

### Task 2: Network Interpretation

To ensure transparency and better understand the model's decisions, we employed **Class Activation Maps (CAM)**. CAM allows us to visualize the regions of the image that were most influential in the model's classification.

- **Layer-wise Activation Visualization**: By visualizing activations across different layers, we gained insights into what features the network learned at various stages.
- **Challenges**: Implementing CAM posed some computational challenges, but it allowed us to refine the model by better understanding misclassifications.

### Task 3: Clean/Dirty Image Transformation

Using a **CycleGAN**, we transformed dirty urban scenes into clean ones. This innovative use of generative adversarial networks (GANs) demonstrates how AI can visualize the potential for cleaner urban environments.

- **CycleGAN Training**: Trained the model to generate clean city images from dirty ones without paired datasets.
- **Successes and Failures**: While some transformations were highly successful, others struggled due to complex waste patterns or lighting conditions.

## Conclusion

This project successfully developed a CNN model for classifying urban scenes and utilized CycleGAN for image enhancement. By understanding AI decisions through CAM, we improved our model's performance and showcased the potential for AI to contribute to urban cleanliness efforts.
