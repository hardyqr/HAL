# VSE-HAL
Code release for "Improved Text-Image Matching by Mitigating Visual Semantic Hubs" [\[arxiv\]](https://arxiv.org/pdf/1911.10097v1.pdf) at AAAI 2020.



## Train

Run `train.py`

#### COCO

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

