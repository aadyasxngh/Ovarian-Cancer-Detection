# Ovarian-Cancer-Detection

This project implements multiple deep learning architectures for the detection and classification of ovarian cancer from medical images. <br>  
The models explored include InceptionV3, ResNet50, MobileNetV2, DPN92, DLA, and enhanced/pruned versions of InceptionV3. <br>  
It also integrates Explainable AI (XAI) techniques to provide better interpretability of the predictions.  

---

**Project Structure**

The project contains the following files:  

- **enhancedinceptionv3.py** → Enhanced InceptionV3 architecture  
- **inceptionv3.py** → Standard InceptionV3 implementation  
- **inceptionv3_pruning20.py** → InceptionV3 with 20% pruning  
- **inceptionv3_pruning30.py** → InceptionV3 with 30% pruning  
- **inceptionv3_pruining40.py** → InceptionV3 with 40% pruning  
- **resnet50.py** → ResNet50 model  
- **mobilenetv2.py** → MobileNetV2 model  
- **dpn92.py** → DPN92 model  
- **DLA.py** → Deep Layer Aggregation model  
- **xai_inceptionv3.py** → Explainable AI integration for InceptionV3  
- **README.md** → Project documentation  


---

## **Features**

- Multiple CNN architectures for cancer detection:  
  - InceptionV3 (base, enhanced, and pruned versions)  
  - ResNet50  
  - MobileNetV2  
  - DPN92  
  - DLA  

- Model pruning experiments for lightweight inference  

- Integration of Explainable AI (XAI) methods for interpretability  
