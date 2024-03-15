# CRL-SemCom-VidCI
This is the official implementation for 'Compression Ratio Learning and Semantic Communications for Video Imaging'  

Test:

For testing learned-ratio methods, run experiment_scripts/train_x.py. 
You can test different models by changing line 72 and 73. 
<pre>
For example, when testing models under logs/24-03-08/24-03-08-MST/MST_adaptive/<b>v_34</b>/checkpoints/model_epoch_0006.pth, line 72/73 should be like,

file = [f for f in os.listdir(f'{dir_name}/v_{34}/checkpoints') if 'model_epoch_0006.pth' in f][0] 

fname = f'{dir_name}/v_{34}/checkpoints/{file}'
<pre>
The average ratio for v_30, v_19, v_10, v_11, v_34 are 0.87, 1.18,  1.3,  1.44,  1.69 respectively.



