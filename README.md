## A Comprehensive Study and Comparison of the Robustness of 3D Object Detectors Against Adversarial Attacks
**Abstract**: Deep learning-based 3D object detectors have made significant progress in recent years and have been deployed in a wide range of applications. It is crucial to understand the robustness of detectors against adversarial attacks when employing detectors in security-critical applications. In this paper, we make the first attempt to conduct a thorough evaluation and analysis of the robustness of 3D detectors under adversarial attacks. Specifically, we first extend three kinds of adversarial attacks to the 3D object detection task to benchmark the robustness of state-of-the-art 3D object detectors against attacks on KITTI and Waymo datasets, subsequently followed by the analysis of the relationship between robustness and properties of detectors. Then, we explore the transferability of cross-model, cross-task, and cross-data attacks. We finally conduct comprehensive experiments of defense for 3D detectors, demonstrating that simple transformations like flipping are of little help in improving robustness when the strategy of transformation imposed on input point cloud data is exposed to attackers.
Our findings will facilitate investigations in understanding and defending the adversarial attacks against 3D object detectors to advance this field.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2212.10230)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Eaphan/Robust3DOD)

## Usage
The code is developed on OpenPCDet v0.5. You can refer to [link](docs/INSTALL.md) to install the pcdet and this [link](docs/GETTING_STARTED.md) to train the detectors.

The adversarial attacks are implemented in three files: [attack.py](attack.py), [remove.py](remove.py), [attach.py](attach.py). You can run the script like this:

```
python attack.py --cfg_file XXX --ckpt XXX 
```


## Citation
If you find this work useful in your research, please consider cite:
```
@article{zhang2022comprehensive,
  title={A Comprehensive Study and Comparison of the Robustness of 3D Object Detectors Against Adversarial Attacks},
  author={Zhang, Yifan and Hou, Junhui and Yuan, Yixuan},
  journal={arXiv preprint arXiv:2212.10230},
  year={2022}
}
```

## Acknowledgement
The code is devloped based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).