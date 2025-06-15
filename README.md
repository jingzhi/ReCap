# ReCap: Better Gaussian Relighting With Cross-Environment Captures (CVPR 2025)
## [Project Page](https://jingzhi.github.io/ReCap/) |  [Paper](https://arxiv.org/abs/2412.07534) | [Dataset](https://drive.google.com/drive/folders/1TH9RXfjrpR7SCcODjzcH47sI5NHieqLR?usp=sharing)


## Setup

#### Tested on Ubuntu 22.04 + CUDA 11.8.
To clone the repo and nvdiffrast:
```bash
git clone https://github.com/jingzhi/ReCap.git
cd submodules
git clone https://github.com/NVlabs/nvdiffrast.git && cd nvdiffrast && git checkout tags/v0.3.1
cd ../..
```

To setup the environment:
```bash
conda env create --file environment.yml
conda activate recap
```
Install imageio HDR support:
```bash
imageio_download_bin freeimage
```

## Reproduce Paper Results
Download RelightObj and HDR maps from [here](https://drive.google.com/drive/folders/1TH9RXfjrpR7SCcODjzcH47sI5NHieqLR?usp=sharing).

Unzip and place `RelightObj` and `high_res_env_maps_1k` under any desired path `<path_to_dataset>`. 

The directory structure should look like this:

```
<path_to_dataset>/RelightObj/
<path_to_dataset>/Environment_Maps/high_res_env_maps_1k/
```

Navigate to the project directory and create a symbolic link to the data directory:

```bash
cd <path_to_project>/ReCap
ln -s <path_to_dataset> data
```

To train and test all objects, run:
```bash
python batch_train_test_script.py
```
## RelightObj Dataset
The [dataset](https://drive.google.com/drive/folders/1TH9RXfjrpR7SCcODjzcH47sI5NHieqLR?usp=sharing) contains 13 objects from [Shiny Blender](https://dorverbin.github.io/refnerf/) and [NeRF Synthetic](https://www.matthewtancik.com/nerf) dataset, rendered under 8 different environments using Blender 3.6.21. 

All light sources and emissive surfaces are removed before rendering.

### Transform files
| File | Content |
|----------|----------|
| transforms_train.json | combined training views (0-199)*8 for all 8 environments|
| transforms_test.json |  combined test views (0-199)*8 for all 8 environments |
| transforms_train_<env_name>.json | training views (0-199) for environment <env_name>|
| transforms_test_<env_name>.json |  test views (0-199) for  environment <env_name> |
| transforms_train_bridge_courtyard.json |  test views (0-99) from bridge and test views (100-199) from courtyard |

### Envmap strength

The provided transform files include the strength of the environment map lighting used during rendering for record-keeping purposes.

The environment map strength is typically set to 1.0, except for the `teapot` and `coffee`. These objects tend to over-expose in the `interior`, `courtyard`, and `snow` environments when the strength is set to 1.0. To align the lighting between training and testing, we apply a scaled environment map when testing these two objects under the mentioned environment maps (see batch_train_test_script.py for details).

### Render extra data

If you wish to render albedo, depth or normal for the provided images or simply render your own object, please refer to [Data Generation](./blender_synth).

## Acknowledgement
We have borrow codes from the following repositories. Many thanks to the authors for sharing their codes.
- [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [GaussianShader](https://github.com/Asparagus15/GaussianShader)

## Citation
If you find this work helpful, please consider citing us:

```latex
@inproceedings{li2025recap,
  title={ReCap: Better Gaussian Relighting with Cross-Environment Captures}, 
  author={Jingzhi Li and Zongwei Wu and Eduard Zamfir and Radu Timofte},
  booktitle={CVPR},
  year={2025},
}
```
