# Ovarian-Cancer-Detection

This project implements multiple deep learning architectures for the detection and classification of ovarian cancer from medical images. <br>  
The models explored include GoogLeNet, ResNet50, MobileNetV2, DPN92, DLA, and enhanced/pruned versions of GoogLeNet. <br>  
It also integrates Explainable AI (XAI) techniques to provide better interpretability of the predictions.  

---

**Project Structure**

The project contains the following files:  

- **enhancedGoogleNet.py** → Enhanced GoogLeNet architecture  
- **googlenet.py** → Standard GoogLeNet implementation  
- **googlenet_pruning20.py** → GoogLeNet with 20% pruning  
- **googlenet_pruning30.py** → GoogLeNet with 30% pruning  
- **googlenet_pruining40.py** → GoogLeNet with 40% pruning  
- **resnet50.py** → ResNet50 model  
- **mobilenetv2.py** → MobileNetV2 model  
- **dpn92.py** → DPN92 model  
- **DLA.py** → Deep Layer Aggregation model  
- **xai_googlenet.py** → Explainable AI integration for GoogLeNet  
- **README.md** → Project documentation  


---

## **Features**

- Multiple CNN architectures for cancer detection:  
  - GoogLeNet (base, enhanced, and pruned versions)  
  - ResNet50  
  - MobileNetV2  
  - DPN92  
  - DLA  

- Model pruning experiments for lightweight inference  

- Integration of Explainable AI (XAI) methods for interpretability  
