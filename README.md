# CodeBase

## ⚙️ Install

- Python 3.8
- PyTorch 1.8.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Clone the github repo and go to the default directory 'codebase_SR'.
git clone https://github.com/yulunzhang/codebase_SR
cd codebase_SR
conda create -n sr_pytorch python=3.8
conda activate sr_pytorch
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python setup.py develop
```

## 💡Tensorboard

After training starts or ends, view the validation (or loss) curve through `Tensorboard`.

Here, we take the example of training a model on a server and viewing Tensorboard locally.

All Tensorboard-log files are in `tb_logger`.

Run the following scripts in the server.

```
ssh -p 22 -L 16006:127.0.0.1:6006 username@remote_server_ip
tensorboard --logdir=tb_logger/xxx --port=6006
```

Then open the web page `http://localhost:6006/` on your local computer.



## 🔗 Contents

1. [Datasets](#datasets)
1. [Models](#models)
1. [Training](#training)
1. [Testing](#testing)

## <a name="datasets"></a>🔎 Datasets

Used training and testing sets can be downloaded as follows:

| Training Set                                                 |                         Testing Set                          |
| :----------------------------------------------------------- | :----------------------------------------------------------: |
| [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images, 100 validation images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) [complete training dataset [DF2K](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link)] | Set5 + Set14 + BSD100 + Urban100 + Manga109 [complete testing dataset [download](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing)] |

Download training and testing datasets and put them into the corresponding folders of `datasets/`. See [datasets](datasets/README.md) for the detail of the directory structure.

## <a name="models"></a>🔎 Models

| Method    | Params | FLOPs | Dataset  |  PSNR   |  SSIM  |                          Model Zoo                           |
| :-------- | :----: | :---: | :------: | :-----: | :----: | :----------------------------------------------------------: |
| FSRCNN-x2 | 22.04K | 5.01G | Urban100 | 27.9280 | 0.8692 | [train_FSRCNN_patch48_batch16_x2](experiments/pretrained_models/train_FSRCNN_patch48_batch16_x2) |
| FSRCNN-x3 | 22.04K | 2.22G | Urban100 | 25.0193 | 0.7622 | [train_FSRCNN_patch48_batch16_x3](experiments/pretrained_models/train_FSRCNN_patch48_batch16_x3) |
| FSRCNN-x4 | 22.04K | 1.25G | Urban100 | 23.6120 | 0.6826 | [train_FSRCNN_patch48_batch16_x4](experiments/pretrained_models/train_FSRCNN_patch48_batch16_x4) |

The performance is reported on Urban100, and output size of FLOPs is 3×1280×720. 

We also provide the **testing results** (log) and **tb_logger** in the folder.



## <a name="training"></a>🔧 Training

- Download [training](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link) (DF2K, already processed) and [testing](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) (Set5, Set14, BSD100, Urban100, Manga109, already processed) datasets, place them in `datasets/`.

- Run the following scripts. The training configuration is in `options/train/`.

  ```shell
  # FSRCNN, x2, input=48x48, 1 GPU
  python basicsr/train.py -opt options/Train/train_FSRCNN_x2.yml
  
  # FSRCNN, x3, input=48x48, 1 GPU
  python basicsr/train.py -opt options/Train/train_FSRCNN_x3.yml
  
  # FSRCNN, x4, input=48x48, 1 GPU
  python basicsr/train.py -opt options/Train/train_FSRCNN_x4.yml
  ```

- The training experiment is in `experiments/`.

## <a name="testing"></a>⚒️ Testing

### 🔨 Test images with HR

- Run the following scripts. The testing configuration is in `options/test/` (e.g., [test_FSRCNN_x4.yml](options/Test/test_FSRCNN_x4.yml)).

  ```shell
  # FSRCNN, x2
  python basicsr/test.py -opt options/Test/test_FSRCNN_x2.yml
  
  # FSRCNN, x3
  python basicsr/test.py -opt options/Test/test_FSRCNN_x3.yml
  
  # FSRCNN, x4
  python basicsr/test.py -opt options/Test/test_FSRCNN_x4.yml
  ```

- The output is in `results/`.

### ⛏️ Test images without HR

- Put your dataset (single LR images) in `datasets/single`. Some test images are in this folder.

- Run the following scripts. The testing configuration is in `options/test/` (e.g., [test_single_x4.yml](options/Test/test_single_x4.yml)).

  ```shell
  # Test on your dataset, x2
  python basicsr/test.py -opt options/Test/test_single_x2.yml
  
  # Test on your dataset, x3
  python basicsr/test.py -opt options/Test/test_single_x3.yml
  
  # Test on your dataset, x4
  python basicsr/test.py -opt options/Test/test_single_x4.yml
  ```

- The output is in `results/`.
