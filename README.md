# VSE-HAL
Code release for "Improved Text-Image Matching by Mitigating Visual Semantic Hubs" [\[arxiv\]](https://arxiv.org/pdf/1911.10097v1.pdf) at AAAI 2020.

## Dependencies
```
nltk==3.4.3
pycocotools==2.0.0
numpy==1.16.4
torch==1.2.0
torchvision==0.4.0
tensorflow-tensorboard==1.5.1
```

## Data
#### MS-COCO
[\[vgg_precomp\]](https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip) <br>
[\[resnet_precomp\]](https://drive.google.com/uc?id=1vtUijEbXpVzNt6HjC6ph8ZzMHRRNms5j&export=download)

#### Flickr30k
[\[vgg_precomp\]](http://www.cs.toronto.edu/~faghri/vsepp/data.tar)

## Train

Run `train.py`

#### MS-COCO

##### w/o global weighting

```bash
python3 train.py \
	--data_path "data/data/resnet_precomp" \
	--vocab_path "data/vocab/" \
	--data_name coco_precomp \
	--batch_size 512 \
	--learning_rate 0.001 \
	--lr_update 8 \
	--num_epochs 13 \
	--img_dim 2048 \
	--logger_name runs/COCO \
	--local_alpha 30.00 \
	--local_ep 0.3
```

##### with global weighting

```bash
python3 train.py \
	--data_path "data/data/resnet_precomp" \
	--vocab_path "data/vocab/" \
	--data_name coco_precomp \
	--batch_size 512 \
	--learning_rate 0.001 \
	--lr_update 8 \
	--num_epochs 13 \
	--img_dim 2048 \
	--logger_name runs/COCO_mb \
	--local_alpha 30.00 \
	--local_ep 0.3 \
	--memory_bank \
	--global_alpha 40.00 \
	--global_beta 40.00 \
	--global_ep_posi 0.20 \
	--global_ep_nega 0.10 \
 	--mb_rate 0.05 \
	--mb_k 250
```

#### Flickr30k

```bash
python3 train.py \
	--data_path "data/data" \
	--vocab_path "data/vocab/" \
	--data_name f30k_precomp \
	--batch_size 128 \
	--learning_rate 0.001 \
	--lr_update 8 \
	--num_epochs 13 \
	--logger_name runs/f30k \
	--local_alpha 60.00 \
	--local_ep 0.7
```

## Evaluate

run `compute_results.py`

#### COCO

```bash
python3 compute_results.py --data_path data/data/resnet_precomp --fold5 --model_path runs/COCO/model_best.pth.tar
```

#### Flickr30k

```bash
python3 compute_results.py --data_path data/data --model_path runs/f30k/model_best.pth.tar
```
#### Trained models
[\[Google Drive\]](https://drive.google.com/drive/folders/1H_EVBFxpYKObNo_CjV0pTaB24A1jWsSF)

## Note
Trained models and codes for replicating results on [SCAN](https://github.com/kuanghuei/SCAN) are coming soon.

## Acknowledgments
This project would not be possible without the open source implementations of [VSE++](https://github.com/fartashf/vsepp) and [SCAN](https://github.com/kuanghuei/SCAN).

## License
[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Reference
Please consider citing our work if you find this repo useful:
```
@article{liu2020hal,
  title={HAL: Improved Text-Image Matching by Mitigating Visual Semantic Hubs},
  author={Liu, Fangyu and Ye, Rongtian and Wang, Xun and Li, Shuaipeng},
  journal={AAAI},
  year={2020}
}
```
