## Study Plan
Research_Material - [Paper_List](topics/Paper_List.md) | Prog - [Programming](topics/programming.md)
| QF-[Quantitative Finance](topics/quantitative_finace.md)

## 2019-12
### Study

* pytorch acceleration
    * [ ] dali ([install guide](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/installation.html))([code](https://github.com/NVIDIA/DALI))
    * [ ] apex ([offical guide](https://nvidia.github.io/apex/index.html))([教程](https://chenyue.top/2019/05/21/%E5%B7%A5%E7%A8%8B-%E4%BA%94-apex%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E5%8A%A0%E9%80%9F/))

### Reading
* 3D Mesh Reconstruction
* 2D Pose Estimation
    * [ ] Pose Neural Fabrics Search ([arXiv](https://arxiv.org/pdf/1909.07068.pdf))([code](https://github.com/yangsenius/PoseNFS))
* Render
    * [ ] Fashion++: Minimal Edits for Outfit Improvement ([ICCV19](https://arxiv.org/abs/1904.09261))([code](https://github.com/facebookresearch/FashionPlus)) : Borrow from [BicycGAN](https://github.com/junyanz/BicycleGAN) and [pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
    

## 2019-11
Main focus: preparing for ICML and ECCV.

Prepare CVPR submissions and supplementary materials.
### Study
* [X] Graph Convolutional Network ([Graph本质解析](https://www.zhihu.com/question/54504471))

### Reading
* 3D Pose Estimation
    * [ ] Bottom-up Higher-Resolution Networks for Multi-Person Pose Estimation ([arXiv](https://arxiv.org/abs/1908.10357))([code](https://github.com/HRNet/Higher-HRNet-Human-Pose-Estimation))
    * [ ] The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation ([arXiv](https://arxiv.org/abs/1911.07524))([zhihu](https://zhuanlan.zhihu.com/p/92525039))
    * [ ] MaskedFusion: Mask-based 6D Object Pose Detection ([arXiv](https://arxiv.org/abs/1911.07771))([code](https://github.com/kroglice/MaskedFusion))   
* Unsupervised 3D Pose Estimation
    * [X] Weakly-Supervised Discovery of Geometry-Aware Representation for 3D Human Pose Estimation ([CVPR'19](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Weakly-Supervised_Discovery_of_Geometry-Aware_Representation_for_3D_Human_Pose_Estimation_CVPR_2019_paper.pdf))
    * [X] Unsupervised Keypoint Learning for Guiding Class-Conditional Video Prediction ([NIPS'19](https://openreview.net/pdf?id=rkl-dNHl8B))
    * [X] Discovery of Latent 3D Keypoints via End-to-end Geometric Reasoning ([NIPS'18](https://arxiv.org/abs/1807.03146))
    * [X] Domes to Drones: Self-Supervised Active Triangulation for 3D Human Pose Reconstruction([NIPS'19](http://papers.nips.cc/paper/8646-domes-to-drones-self-supervised-active-triangulation-for-3d-human-pose-reconstruction.pdf))
    * [X] Unsupervised 3D Pose Estimation with Geometric Self-Supervision
* Graph
    * [X] Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation([IJCAI'18](https://arxiv.org/abs/1804.06055))
    * [X] Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition([AAAI'18](https://arxiv.org/pdf/1801.07455.pdf))
    * [X] Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition ([CVPR'19](https://zpascal.net/cvpr2019/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf))
* NAS
    * [X] ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware ([ICLR19](https://arxiv.org/pdf/1812.00332.pdf))([code](https://github.com/mit-han-lab/ProxylessNAS))
    * [X] FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search ([facebook](https://arxiv.org/abs/1812.03443))([code](https://github.com/AnnaAraslanova/FBNet))
    * [X] Fair DARTS: Eliminating Unfair Advantages in Differentiable Architecture Search ([arXiv]https://arxiv.org/abs/1911.12126)([code](https://github.com/xiaomi-automl/fairdarts))
    * [X] Learning Graph Convolutional Network for Skeleton-based Human Action Recognition by Neural Searching ([arXiv](https://arxiv.org/abs/1911.04131))
* Tracking
    * [ ] You Only Look Once: Unified, Real-Time Object Detection ([arXiv](https://arxiv.org/abs/1506.02640))
* Detection
    * [ ] EfficientDet: Scalable and Efficient Object Detection ([arXiv](https://arxiv.org/abs/1911.09070))([zhihu](https://zhuanlan.zhihu.com/p/93241232))
* Render
    * [ ] DeepFovea: Neural Reconstruction for Foveated Rendering and Video Compression using Learned Statistics of Natural Videos ([Facebook Reality Labs](https://research.fb.com/wp-content/uploads/2019/11/DeepFovea-Neural-Reconstruction-for-Foveated-Rendering-and-Video-Compression-using-Learned-Statistics-of-Natural-Videos.pdf?))
    * [ ] Animating Landscape: Self-Supervised Learning of Decoupled Motion and Appearance for Single-Image Video Synthesis ([TOG'19](http://www.cgg.cs.tsukuba.ac.jp/~endo/projects/AnimatingLandscape/animating_landscape_siga19.pdf))([project](http://www.cgg.cs.tsukuba.ac.jp/~endo/projects/AnimatingLandscape/))([code](https://github.com/endo-yuki-t/Animating-Landscape))
## 2019-10
Work hard for CVPR2020 and PRCV2019 Challenge workshop (**Rank 8th**).

## 2019-9
Start for PHD in Vision and Learning Lab supervised by Bingbing Ni and Wenjun Zhang.

## Before 2019
* ILSVRC 2015: Classification+localization with additional training data (**Rank 1st**). 
* ILSVRC 2016: Object detection/tracking from video with additional training data (**Rank 1st**). 
* ILSVRC 2016: Object detection from video with provided/additional training data (**Rank 1st**). 
* ILSVRC 2017: Object detection with provided/additional training data (**Rank 1st**). 
* DAVIS Challenge 2016（**just in experiments**）: Unsupervised Video Segmentation (When i was intern in MSRA supervised by  Yan Lv and Xiulian Peng in 2018) (**Rank 1st**)