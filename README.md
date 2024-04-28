# CRL-SemCom-VidCI
This is the official implementation for 'Compression Ratio Learning and Semantic Communications for Video Imaging'  [paper](https://arxiv.org/pdf/2310.06246)

Test:

For testing learned-ratio methods, run experiment_scripts/train_x.py. 
You can test different models by changing line 72 and 73. 
<pre>
For example, when testing models under logs/24-03-08/24-03-08-MST/MST_adaptive/<b>v_34</b>/checkpoints/<b>model_epoch_0006.pth</b>, line 72/73 are,

file = [f for f in os.listdir(f'{dir_name}/<b>v_{34}</b>/checkpoints') if '<b>model_epoch_0006.pth</b>' in f][0] 

fname = f'{dir_name}/<b>v_{34}</b>/checkpoints/{file}'
</pre>

The average ratio for v_30, v_19, v_10, v_11, v_34 are 0.87, 1.18,  1.3,  1.44,  1.69 respectively.

For testing fixed-ratio methods, also run experiment_scripts/train_x.py. but make the following changes:
1. Change line 198 from parser.add_argument('--exp_name', type=str, default='MST_adaptive') to parser.add_argument('--exp_name', type=str, default='MST_fixed')
2. still change line 72/73 to test models under logs/24-03-08/24-03-08-MST/MST_fixed/v_1 or v_2 or v_3 or v_4
3. open  line 353 in shutters/shutters_adaptive5_nomask.py and for testing v_1, v_2, v_3, v_4, action=1*, 2*, 3*, 4* torch.ones_like(action), respectively.

Besides, we made a mistake for fixed ratio method when average ratio is 1 and mask B is not used. The PSNR should be 32.3 not 29.54. 29.54 is the performance when mask B is used. I drew the picture in a hurry so made a mistake here.

## Additional information
If you find the source code is useful for your research, please cite our paper:  
@article{zhang2023compression,
  title={Compression Ratio Learning and Semantic Communications for Video Imaging},
  author={Zhang, Bowen and Qin, Zhijin and Li, Geoffrey Ye},
  journal={arXiv preprint arXiv:2310.06246},
  year={2023}
}



