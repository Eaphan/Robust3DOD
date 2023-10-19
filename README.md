## A Comprehensive Study of the Robustness for LiDAR-based 3D Object Detectors against Adversarial Attacks
**Abstract**: Recent years have witnessed significant advancements in deep learning-based 3D object detection, leading to its widespread adoption in numerous applications. As 3D object detectors become increasingly crucial for security-critical tasks, it is imperative to understand their robustness against adversarial attacks. This paper presents the first comprehensive evaluation and analysis of the robustness of LiDAR-based 3D detectors under adversarial attacks. Specifically, we extend three distinct adversarial attacks to the 3D object detection task, benchmarking the robustness of state-of-the-art LiDAR-based 3D object detectors against attacks on the KITTI and Waymo datasets. We further analyze the relationship between robustness and detector properties. Additionally, we explore the transferability of cross-model, cross-task, and cross-data attacks. Thorough experiments on defensive strategies for 3D detectors are conducted, demonstrating that simple transformations like flipping provide little help in improving robustness when the applied transformation strategy is exposed to attackers. Finally, we propose balanced adversarial focal training, based on conventional adversarial training, to strike a balance between accuracy and robustness. Our findings will facilitate investigations into understanding and defending against adversarial attacks on LiDAR-based 3D object detectors, thus advancing the field.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2212.10230)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Eaphan/Robust3DOD)

## Usage
The code is developed on OpenPCDet v0.5. You can refer to [link](docs/INSTALL.md) to install the pcdet and this [link](docs/GETTING_STARTED.md) to train the detectors.

The adversarial attacks are implemented in three files: [attack.py](attack.py), [remove.py](remove.py), [attach.py](attach.py). You can run the script like this:

```
python attack.py --cfg_file XXX --ckpt XXX 
```


## Citation
If you find this work useful in your research, please consider citing:
```
@article{zhang2023comprehensive,
  title={A Comprehensive Study of the Robustness for LiDAR-based 3D Object Detectors against Adversarial Attacks},
  author={Zhang, Yifan and Hou, Junhui and Yuan, Yixuan},
  journal={International Journal of Computer Vision},
  year={2023}
}
```

## Acknowledgement
The code is developed based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).
