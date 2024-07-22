# CRL-SemCom-VidCI
This is the official implementation for 'Compression Ratio Learning and Semantic Communications for Video Imaging'  [paper]([https://ieeexplore.ieee.org/abstract/document/10539255])

**Update 22/07/2024** We have uploaded the complete training/testing codes for compression ratio learning.

## Compression Ratio Learning

<b>Test</b>:

For testing learned-ratio methods, run experiment_scripts/project_test_x.py.
<br>
You can test different models by changing line 72 and 73 of project_test_x.py. 
<pre>
For example, when testing models under logs/24-03-08/24-03-08-MST/MST_adaptive/<b>v_34</b>/checkpoints/<b>model_epoch_0006.pth</b>, line 72/73 are,

file = [f for f in os.listdir(f'{dir_name}/<b>v_{34}</b>/checkpoints') if '<b>model_epoch_0006.pth</b>' in f][0] 

fname = f'{dir_name}/<b>v_{34}</b>/checkpoints/{file}'
</pre>

The average ratio for v_30, v_19, v_10, v_11, v_34 are 0.87, 1.18,  1.3,  1.44,  1.69 respectively. 
<br>
For testing fixed-ratio methods, also run experiment_scripts/project_test_x.py. but pls make the following changes:
<br>
1. Change line 198 from parser.add_argument('--exp_name', type=str, default='MST_adaptive') to parser.add_argument('--exp_name', type=str, default='MST_fixed')
2. still change line 72/73 to test models under logs/24-03-08/24-03-08-MST/MST_fixed/v_1 or v_2 or v_3 or v_4
3. open  line 366 in shutters/shutters_adaptive5_nomask.py and for testing v_1, v_2, v_3, v_4, action=1*, 2*, 3*, 4* torch.ones_like(action), respectively.
<be>

Besides, we made a mistake when plotting the figure for the fixed ratio method when the average ratio is 1 and mask B is not used. The PSNR should be 32.3 not 29.54. 29.54 is the performance when mask B is used. 
<br>

<b>Train</b>:
<br>
For training learned-ratio methods, run experiment_scripts/train_x.py.
<br>
When training learned-ratio methods, we recover the video reconstruction network in fixed-ratio methods with action=4 by default.
<br>
To adjust the average sampling rate, adjust the parameter in line 177 of train_x.py. As a reference, when 0.05 is used, the average ratio is about 3.25; when 1 is used, the average ratio is about 1.82.
<br>
## Additional information
If you find the source code is useful for your research, please cite our paper:  
@ARTICLE{10539255,
  author={Zhang, Bowen and Qin, Zhijin and Li, Geoffrey Ye},
  journal={IEEE Journal of Selected Topics in Signal Processing}, 
  title={Compression Ratio Learning and Semantic Communications for Video Imaging}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  keywords={Image coding;Semantics;Sensors;Imaging;Time measurement;Optics;Image reconstruction},
  doi={10.1109/JSTSP.2024.3405853}}


