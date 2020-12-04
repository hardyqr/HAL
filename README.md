# VSE-HAL
Code release for **HAL: Improved Text-Image Matching by Mitigating Visual Semantic Hubs** [\[arxiv\]](https://arxiv.org/pdf/1911.10097v1.pdf) at AAAI 2020.

```bibtex
@inproceedings{liu2020hal,
  title={{HAL}: Improved text-image matching by mitigating visual semantic hubs},
  author={Liu, Fangyu and Ye, Rongtian and Wang, Xun and Li, Shuaipeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={11563--11571},
  year={2020}
}
```

Upgrade your text-image matching model with a few lines of code:
```python
class ContrastiveLoss(nn.Module):
	...
	def forward(self, im, s, ...):
        	bsize = im.size()[0]
        	scores = self.sim(im, s)
		...
		tmp  = torch.eye(bsize).cuda()
		s_diag = tmp * scores
		scores_ = scores - s_diag
		...
		S_ = torch.exp(self.l_alpha * (scores_ - self.l_ep))
		loss_diag = - torch.log(1 + F.relu(s_diag.sum(0)))

        	loss = torch.sum( \
                	torch.log(1 + S_.sum(0)) / self.l_alpha \
                	+ torch.log(1 + S_.sum(1)) / self.l_alpha \
                	+ loss_diag \
                	) / bsize

        return loss
```


## Dependencies
```
nltk==3.4.5
pycocotools==2.0.0
numpy==1.18.1
torch==1.5.1
torchvision==0.6.0
tensorboard_logger == 0.1.0
```

## Data
#### MS-COCO
[\[vgg_precomp\]](https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip) <br>
[\[resnet_precomp\]](https://drive.google.com/uc?id=1vtUijEbXpVzNt6HjC6ph8ZzMHRRNms5j&export=download)

#### Flickr30k
[\[vgg_precomp\]](http://www.cs.toronto.edu/~faghri/vsepp/data.tar)

## Train

Run `train.py`.

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

Run `compute_results.py`.

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
This project would be impossible without the open source implementations of [VSE++](https://github.com/fartashf/vsepp) and [SCAN](https://github.com/kuanghuei/SCAN).

## License
[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
