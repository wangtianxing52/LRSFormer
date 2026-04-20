# LRSFormer: A Redundancy-Suppressed Transformer for Large-Scale Hyperspectral Image Semantic Segmentation

## Dataset

Owing to its wide spatial extent and diverse classification scenarios, the [WHU-OHS](https://github.com/zjjerica/WHU-OHS-Pytorch) dataset provides a challenging testbed for large-scale hyperspectral image semantic segmentation.

The [WHU-Hi](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm) dataset is an important benchmark for hyperspectral image analysis, benefiting from its high spatial resolution and finely defined categories.

Featuring complex urban land-cover distributions and a multi-sensor acquisition setting, the [Houston 2018](https://www.grss-ieee.org/community/technical-committees/2018-ieee-grss-data-fusion-contest/) dataset is widely used for hyperspectral image semantic segmentation research.

## Requirements

The code is developed and tested under the following software environment:

```text
timm==1.0.9
tqdm==4.66.1
numpy==1.23.5
einops==0.7.0
ptflops==0.6.9
python==3.9.19
matplotlib==3.7.2
torch==1.12.0+cu113
torch-summarry==1.4.5
mmsegmentation==1.2.2