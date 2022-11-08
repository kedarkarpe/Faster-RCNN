# Faster-RCNN

# Introduction
In this project you will implement some of the components of MaskRCNN,  an  algorithm  that addresses the task of instance seg-mentation, which combines object detection and semantic segmentation into a per-pixel object detection framework.

<div><img src="https://github.com/LukasZhornyak/CIS680_files/raw/main/HW2/fig2_1.png"/></div>
<center>Figure 1: This is a demo of what object detection does. The color indicates different semantic class.</center>  

# Region-Proposal Network
Region Proposal Networks (RPNs) are ”attention mechanisms” for the object detection task, performing a crude but inexpensive first estimation of where the bounding boxes of the objects should be. They were first proposed as a way to address the issue of expensive greedy algorithms like Selective Search, opening new avenues to end-to-end object detection tasks. They work through classifying the initial anchor boxes into object/background and refine the coordinates for the boxes with objects. Later, these boxes will be further refined and tightened by the instance segmentation heads as well as classified in their corresponding classes. The architecture for the RPN and the later refinement of the proposals is shown below:
