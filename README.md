# Classifying (using VGG16) and Captioning (using BLIP) Damaged Aircrafts

## Project Overview
This project automates aircraft damage detection (dent vs. crack) using VGG16 transfer learning for classification and BLIP for image captioning/summarization. It achieves high accuracy on Roboflow Aircraft Dataset; ideal for aviation safety inspections. It replaces manual checks with AI—classify damage from images and generate descriptive captions like "a picture of a plane" or summaries detailing fuselage/engine issues.

## Dataset
**Source:** Roboflow Aircraft Damage Dataset (CC BY 4.0)  
**Splits:** Train (300 imgs), Valid (96), Test (50)  
**Classes:** dent, crack  
**Preprocessed:** Resized to 224x224, rescaled 1./255

## Model Architecture 

**Part 1: Classification (VGG16 Feature Extractor)**
- Base: VGG16 (ImageNet weights, no top), frozen layers
- Custom Head:
  - Flatten → Dense(512, ReLU) → Dropout(0.3) → Dense(512, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- Optimizer: Adam (lr=0.0001)
- Loss: Binary Crossentropy
- Training: 5 epochs, batch=32

Results (from training history):
| Epoch | Train Acc | Val Acc | Train Loss | Val Loss |
| ----- | --------- | ------- | ---------- | -------- |
| 1     | 53.7%     | 60.4%   | 0.720      | 0.634    |
| 2     | 72.3%     | 69.8%   | 0.554      | 0.582    |
| 3     | 77.7%     | 71.9%   | 0.474      | 0.551    |
| 4     | 81.0%     | 66.7%   | 0.414      | 0.649    |
| 5     | 87.0%     | 69.8%   | 0.321      | 0.509    |

**Part 2: Captioning (BLIP)**
- Model: Salesforce/blip-image-captioning-base
- Tasks: Caption ("a picture of a plane") or Summary ("detailed photo showing engine of Boeing 747")
- Custom Keras Layer: BlipCaptionSummaryLayer for easy integration

Example Outputs:
- Caption: "this is a picture of a plane"
- Summary: "this is a detailed photo showing the damage to the fuselage of the aircraft"

## Results
The VGG16 classification model achieved 87.0% training accuracy and 69.8% validation accuracy after 5 epochs, with training loss dropping from 0.72 to 0.32. The BLIP captioning model successfully generated descriptive outputs like "this is a picture of a plane" for captions and "detailed photo showing damage to the fuselage" for summaries.

## Tech Stack
- tensorflow-cpu==2.17.1
- transformers==4.38.2
- torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu
- pandas==2.2.3 pillow==11.1.0 matplotlib==3.9.2

## How to Run

Clone the repository:

    git clone https://github.com/jsharonjr/classifying-and-captioning-damaged-aircrafts

Navigate to the project folder:

    cd classifying-and-captioning-damaged-aircrafts
    
Install dependencies:

    pip install tensorflow numpy matplotlib
    
Open and run the notebook:

    Classification_and_Captioning.ipynb

## License
Distributed under the MIT License. See [LICENSE](https://github.com/jsharonjr/classifying-and-captioning-damaged-aircrafts/blob/main/LICENSE) for more information.
