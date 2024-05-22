# HackTrick 2024

Welcome to HackTrick 2024! This README provides an overview of the game and its components. For detailed instructions and documentation, please refer to the accompanying documents in the repository.

## Introduction

HackTrick 2024 is a steganography-based hackathon where teams compete in two phases, playing as both Fox and Eagle roles. The game revolves around sending and intercepting secret messages concealed within images.

## Game Description

In this year's hackathon, contestants take on the roles of Fox and Eagle. The Fox attempts to encode and transmit secret messages using steganography techniques, while the Eagle aims to intercept and decode these messages. The game environment is designed to challenge participants' skills in cryptography and deception.

## Game Environment

Every team receives a unique team ID to interact with the server using APIs provided in the accompanying documentation. This README provides an overview of the game, while detailed instructions and strategies can be found in the associated documents.

### Fox

As the primary player, the Fox's objective is to encode and send secret messages while outsmarting the Eagle. The Fox operates in chunks, sending messages through three channels using steganography techniques.

#### Riddles

To enhance the gameplay, teams can solve riddles to earn Fake messages, which can be used to deceive the Eagle. Riddles are categorized by difficulty and domain, offering various rewards based on their complexity.

#### Steganography: LSB

The Fox utilizes the Least Significant Bit (LSB) method to encode messages within images. This method allows for discrete communication by altering the LSB of each pixel in an image.

#### Scoring Function

The Fox's performance is evaluated based on message transmission efficiency, time management, and Fake message utilization. The scoring function considers factors such as message accuracy and budget management.

### Eagle

The Eagle's role involves intercepting and decoding messages sent by the Fox. By analyzing footprints and identifying genuine messages, the Eagle aims to outsmart the Fox and prevent deceptive transmissions.

#### Footprints

Footprints provide visual representations of audio files transmitted through each channel. The Eagle must correctly identify genuine messages to decode and score points.

#### Requesting The Message

The Eagle requests messages based on identified footprints and decodes them using the LSB method. Time efficiency and message accuracy are crucial for maximizing scores.

#### Scoring Function

The Eagle's score depends on message decoding accuracy, time efficiency, and the ability to dodge Fake messages. The scoring function evaluates performance in identifying and intercepting messages.

### Eagle Model

Teams have access to two datasets containing Real and Fake message footprints for training their Eagle models. These datasets enable teams to develop AI-driven approaches for message interception.

#### Overview

The Eagle model is a deep learning model designed to intercept and decode messages sent by the Fox during the Hacktrick 2024 game. It employs convolutional neural network (CNN) layers followed by dense layers for classification, enabling it to analyze audio footprints, identify the presence of real messages, and decode them for further analysis.

#### Model Architecture

The Eagle model architecture consists of the following layers:

- Input Layer: BatchNormalization layer to preprocess input data.
- Convolutional Layers: Several Conv2D layers with ReLU activation functions and MaxPooling2D layers to extract important features from the input spectrogram.
- Flatten Layer: Flattens the output from the convolutional layers to prepare it for the dense layers.
- Dense Layers: Two fully connected dense layers with ReLU activation and a dropout layer to prevent overfitting.
- Output Layer: A single neuron with a sigmoid activation function to output the probability of a real message being present.

#### Training

The model is trained using the Adam optimizer and binary cross-entropy loss function. Early stopping and model checkpointing callbacks are employed to monitor validation loss and save the best model during training.

#### Data Preparation

The training and testing data consist of audio footprints categorized as real and fake messages. These footprints are preprocessed, and any infinite values are replaced with zeros to ensure numerical stability during training.

#### Evaluation

The model's performance is evaluated based on its ability to correctly classify real and fake messages. Accuracy and loss metrics are used to assess the model's performance on both the training and testing datasets.

#### Future Improvements

To enhance the model's performance, additional data augmentation techniques, hyperparameter tuning, or more sophisticated network architectures could be explored.

### Riddles

Riddles serve as optional side quests for teams to earn Fake messages. Categories include Security, Computer Vision, Machine Learning, and Problem Solving, each offering different rewards based on difficulty.

### APIs

Teams interact with the game server using APIs provided in the accompanying documentation. APIs facilitate message transmission, riddle solving, and score tracking, ensuring smooth gameplay.

For more detailed instructions, API documentation, and game strategies, please refer to the accompanying documents in the repository. Good luck and happy hacking!
