# What is this?
This repo contains slides and datasets used in the 2023 course [Introduction to deep learning for biologists](https://www.physalia-courses.org/courses-workshops/course67/). Material is released during the course week and will stay online a few months after the course.

Exercises are in the form of [jupyter notebooks](https://jupyter.org/) and can be either downloaded and run locally or explored online via platforms such as [Binder](https://mybinder.org/), [Google Colab](https://colab.research.google.com/) (a free Google account is required) or [Kaggle Notebooks](https://www.kaggle.com/notebooks) (a free Kaggle account plus phone verification is required).

## Python - basic skills

This section moved to the code-only repo [bioinformateachers](https://github.com/ne1s0n/bioinformateachers), which supports the bioinformatics-oriented educational website [The Bioinformateachers](https://bioinformateachers.github.io/).

## Slides

[Here you find](slides) all the slides used in the course (we upload them at the end of each day).

## Codes

All codes used during the lessons plus some extra exercises and solutions are found in the
various lab folders (e.g. [lab_day1](lab_day1/)).

## Course Timetable

**Day 1**
- Lecture 0: Introducing the course,the instructors and the participants [day1_block00 Introductions](slides/)
- Lab 1: Introduction to Jupyter notebooks and Python libraries [day1_code00 basic python](lab_day1/day1_code00_basic_python_[EXERCISE].ipynb)
- Lecture 1: Introduction to deep learning [day1_block01 Introduction to DL](slides/)
- Lecture 2 + Lab 2: MNIST data problem
  -  [day1_block02 A DL-NN for image recognition](slides/)
  -  [keras.mnist_train.py](lab_day1/keras.mnist_train.py) [test.py](lab_day1/keras.mnist_test.py)
  -  [day1_code01 keras_MNIST](lab_day1/day1_code01%20keras_MNIST.ipynb)
- Lecture 3: Supervised learning [day1_block03 supervised_learning](slides)
- Lecture 4: Building blocks 1 [day1_block04 Building blocks of DL](slides)
- Lecture 5: Introduction to Keras [day1_block05 Keras](slides)
- Lab 3: Play with keras + Tensorflow Playground
  - [day1_code02 keras basics](lab_day1/day1_code02_keras_basics_[EXERCISE].ipynb)
  - [Tensorflow playground](https://playground.tensorflow.org/)
  - [day1_block06 day1 wrap-up](slides)

**Day 2**
- Lab 2 recap: the MNIST functions explained [day2_code00 keras_MNIST_detailed](lab_day2/day2_code00_keras_MNIST_detailed.ipynb)
- Lecture 6: Logistic regression	[day2_block01 logistic regression and binary classification](slides)
- Lab 4: Hands-on logistic regression [day2_code01 logistic regression iris [EXERCISE]](lab_day2/day2_code01 logistic regression iris [EXERCISE].ipynb)
- Lecture 7a: From logistic regression to neural networks [day2_block02 Neural networks models](slides)
- Lecture 7b: Deep neural networks	[day2_block02 Neural networks models](slides)
- Lab 5: Hands-on neural networks models [day2_code02 keras shallow neural networks](lab_day2/day2_code02 keras shallow neural networks.ipynb)
- Students exercise: Neural networks models [day2_code04 neural networks [EXERCISE]](lab_day2/day2_code03_neural_networks_[EXERCISE].ipynb)
- Quick snippet: Neural Networks for feature selection [demo](lab_day2/day2_code04_feature_selection_[PILL].ipynb)

**Day 3**
- Lecture 8: Multiclass classification and softmax regression [day3_block01 Multiclass classification](slides)
- Lab 6: Multiclass classification and softmax regression [day2_code03 keras multiclass classification](lab_day3/day3_code01_keras_multiclass_classification.ipynb)
- Lecture 9: Cross-validation	[day3_block02 Crossvalidation](slides)
- Lab 7: Practical cross-validation with deep learning [day3_code02 heart disease crossv.ipynb](lab_day3)
- Lecture 10: Building blocks 2 [day3_block03 Building blocks of DL #2](slides)
- Lab 8: Looking inside convolutions [day3_code03 inside convolution.ipynb](lab_day3)
- Exercise: Deep learning models [day3_code03 heart disease crossv [EXERCISE].ipynb](lab_day3)
- Day 3 wrap-up discussion [day3_block04 day 3 wrap-up](slides)

**Day 4**
- Lecture 11 + Lab 9: Data generators and data augmentation
    - [day4_block01 Data generators and data augmentation](slides)
    - [day4_code01 data augmentation [EXERCISE]](lab_day4)`
- Lecture 12: RNN theory - part 1 [day4_block02 RNN models #1](slides)
- Lab 10: [RNN models](lab_day4) + [time series data](lab_day5) 
- Lecture 13 Recipe for a good project	[day4_block03 The recipe for a good project](slides)
- Lab 11: Recap exercise [day4_code03 chest x rays (data augm, regul) [EXERCISE]](lab_day4)
- Lab 12: Under/Over fitting [day4_code04_under_over_fitting.ipynb](lab_day4) and Double descent	[day4_block04 Bias-Variance Trade-off and double descent](slides)
- Lab 13: Deep learning for regression [day4_code05 keras regression [EXERCISE]](lab_day4)
- Lab 14: Semi-automated hyperparameters-tuning [day4_block05 day 4 wrap-up](lab_day4)
  
**Day 5**
- Lecture 14 Transfer learning [day5_block01 Architectures and transfer learning](slides) + [day5_code01_chest_x_rays_models.ipynb](lab_day5)
- Lecture 15: RNN theory - part 2 [day5_block02 RNN models #2](slides)
- Lab 15: RNN lab
  - [day4_code02 RNN-models](lab_day4)
  - [day5_code02 RNN-time series data](lab_day5)
  - [day5_code03 RNN-text data](lab_day5)
- Lecture 16: Segmentation + demo
  - [day5_block03 Image Segmentation](slides)
  - [day5_code04 segmentation metrics](lab_day5)
  - [day5_code05 U-Net](lab_day5)
- Final Quiz
- Wrap-up discussion
