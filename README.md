# DomainDrop: Suppressing Domain-Sensitive Channels for Domain Generalization

## Requirements

* Python == 3.7.3
* Pytorch == 1.8.1
* Cuda == 10.1
* Torchvision == 0.4.2
* Tensorflow == 1.14.0
* GPU == RTX 2080Ti

## DataSets
Please download PACS dataset from [here](https://drive.google.com/drive/folders/0B6x7gtvErXgfUU1WcGY5SzdwZVk?resourcekey=0-2fvpQY_QSyJf2uIECzqPuQ).
Make sure you use the official train/val/test split in [PACS paper](https://openaccess.thecvf.com/content_iccv_2017/html/Li_Deeper_Broader_and_ICCV_2017_paper.html).
Take `/data/DataSets/` as the saved directory for example:
```
images -> /data/DataSets/PACS/kfold/art_painting/dog/pic_001.jpg, ...
splits -> /data/DataSets/PACS/pacs_label/art_painting_crossval_kfold.txt, ...
```
Then set the `"data_root"` as `"/data/DataSets/"` and `"data"` as `"PACS"` in both `train_domain.py` and `train.sh`.

## Training
For training the model, please set the `"result_path"` where the results are saved in both `train_domain.py` and `train.sh`.
Then simply running the code to train a ResNet-18:
```
python train_domain.py --target [domain_index] --device [GPU_index]
```
The `domain_index` denotes the index of target domain, and `GPU_index` denotes the GPU device number.
```
domain_index: [0:'photo', 1:'art_painting', 2:'cartoon', 3:'sketch']
```
Or run the `train.sh` directly.

## Evaluation



To evaluate the performance of the models, you can download the models trained  on PACS as below:

Target domain  | Photo | Art | Cartoon | Sketch |
:----:  | :----: | :----: | :----: | :----: |
Acc(%) | 96.71 | 84.91 | 81.19 | 84.32 |
models | [download](https://drive.google.com/drive/folders/1N63V8HxLXRl94GZgllQHTrxWrqH2-GDl?usp=drive_link) | [download](https://drive.google.com/drive/folders/1zA9smbTRExm6FSu5WpfI0tmx93uonjuk?usp=drive_link) | [download](https://drive.google.com/drive/folders/1jJW4q-aUVsNcUeiE8wKbv0zuzK5f3aJA?usp=drive_link) | [download](https://drive.google.com/drive/folders/1x-33N1mtAJP08sT5dqZX53Y8B_8_Vify?usp=drive_link) |


Please set the `--eval = 1` and `--eval_model_path` as the saved path of the downloaded models.  *e.g.*,  `/trained/model/path/photo/model.pt`. Then you can simple run:

```
python train_domain.py --target [domain_index] --device [GPU_index] --eval 1 --eval_model_path '/trained/model/path/photo/model.pt'
```

## Citations
```
@inproceedings{guo2023domaindrop,
  title={DomainDrop: Suppressing Domain-Sensitive Channels for Domain Generalization},
  author={Guo, Jintao and Qi, Lei and Shi, Yinghuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

## Acknowledgement
Part of our code is derived from the following repository.
* [MMLD](https://github.com/mil-tokyo/dg_mmld): "Domain Generalization Using a Mixture of Multiple Latent Domains", AAAI 2020

We thank to the authors for releasing their codes. Please also consider citing their work.


