## 🔬 Key Concepts

- **Left Ventricle Segmentation**: Accurate identification of the LV in echocardiograms is crucial for cardiac function assessment.
- **UNet Architecture**: A baseline encoder–decoder model for biomedical image segmentation.
- **Pretrained Encoders**: Incorporating backbones like **ResNet50** to enhance feature extraction within the UNet framework.
- **Evaluation Metrics**:
  - **Dice Coefficient** and **IoU** for spatial overlap.
  - **Precision** and **Recall** to understand model behavior in terms of false positives and false negatives.

## 📓 Notebooks

- **Exploratory Data Analysis**: Understand the dataset characteristics and visualize sample frames.  
  [Open in Colab](https://colab.research.google.com/drive/1EQGr7LMHNdov_Vxuk5V9x-JSpUDJwdkm#scrollTo=iowHamIZnI-2)

- **UNet (Baseline)**: 
  Implementation and training of a standard UNet model from scratch.  
  [Open in Colab](https://colab.research.google.com/drive/16NB2oPaZb5Unyc4SEIVSiahU_8pzIMsq#scrollTo=nGZIpARTNMhe)

- **ResNet50-UNet**: A UNet model using a **ResNet50** encoder pretrained on ImageNet, compared for improved segmentation quality.  
  [Open in Colab](https://colab.research.google.com/drive/1YvaR7KLMvMowcxsjpiWhKF3xVVUjL8P9?usp=share_link)

