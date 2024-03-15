import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import os
import shutters.shutter_utils as shutils
import torch.nn.functional as F
import scipy.io as sio
device = 'cuda:0'

repo_dir = '/home/cindy/PycharmProjects/coded-deblur-publish/shutters/shutter_templates'


# classic residual block
class RB(nn.Module):
    def __init__(self, nf, bias, kz=3):
        super(RB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, kz, padding=kz // 2, bias=bias), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kz, padding=kz // 2, bias=bias),
        )

    def forward(self, x):
        return x + self.body(x)


class D(nn.Module):
    def __init__(self, img_nf):
        super(D, self).__init__()
        bias, block, nb, mid_nf = True, RB, 3, 32
        shu=nn.PixelUnshuffle(2)
        out=nn.PixelShuffle(2)
        conv = lambda in_nf, out_nf: nn.Conv2d(in_nf, out_nf, 3, padding=1, bias=bias)
        conv_transpose1= lambda in_nf, out_nf: nn.ConvTranspose2d(in_nf, out_nf, kernel_size=2, stride=2, padding=0)
        conv_transpose= lambda in_nf, out_nf: nn.ConvTranspose2d(in_nf, out_nf, kernel_size=8, stride=4, padding=2)
        self.body = nn.Sequential(conv(img_nf, mid_nf), *[block(mid_nf, bias) for _ in range(nb)], conv_transpose(mid_nf, 5))
        

    def forward(self, x):
        t=self.body(x)
        
        return t.softmax(dim=1)

class Shutter:
    def __new__(cls, shutter_type, block_size, test=False, resume=False, model_dir='', init='even'):
        cls_out = {

            'lsvpe': LSVPE,
        }[shutter_type]

        return cls_out(block_size, test, resume, model_dir, init)


class ShutterBase(nn.Module):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__()
        self.block_size = block_size
        self.test = test
        self.resume = resume
        self.model_dir = os.path.dirname(model_dir) # '21-11-14/21-11-14-net/short'

    def getLength(self):
        raise NotImplementedError('Must implement in derived class')

    def getMeasurementMatrix(self):
        raise NotImplementedError('Must implement in derived class')

    def forward(self, video_block, train=True):
        raise NotImplementedError('Must implement in derived class')

    def post_process(self, measurement, exp_time, test):
        measurement = torch.div(measurement, exp_time)       # 1 1 H W
        measurement = add_noise(measurement, exp_time=exp_time, test=test)
        measurement = torch.div(measurement, exp_time)*8.
        measurement = torch.clamp(measurement, 0, 1)
        # m=measurement.detach().cpu().numpy()
        return measurement

    def count_instances(self, lengths, counts):
        flattened_lengths = lengths.reshape(-1, ).type(torch.int8)
        total_counts = torch.bincount(flattened_lengths).cpu()
        for k in range(1, len(total_counts)):
            counts[k] = total_counts[k]
        return counts



class LSVPE(ShutterBase):
    def __init__(self, block_size, test=False, resume=False, model_dir='', init='even'):
        super().__init__(block_size, test, resume, model_dir, init)
        self.block_size = block_size



        rand_init = torch.ones((1,1, self.block_size[1]//4, self.block_size[2]//4), dtype=torch.float32)
        self.rand_init_new = nn.Parameter(rand_init, requires_grad=True).to(device)


        self.model=D(img_nf=1)




        # self.rand_end=rand_end
        a = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                      2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
                      3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
                      4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4], dtype=np.float32)
        self.time_range = torch.from_numpy(a)[None, :, None, None].to(device)
        self.time_range = self.time_range.repeat(1, 1, self.block_size[1], self.block_size[2])

        # self.time_range_small = torch.from_numpy(a)[None, :, None, None].to(device)
        # self.time_range_small = self.time_range_small.repeat(1, 1, self.block_size[1]//4, self.block_size[2]//4)

        self.total_steps = 0
        self.lengths = torch.zeros((self.block_size[1], self.block_size[2]))
        self.total_steps = 0
        self.counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        self.mean_=0

        self.up = nn.Upsample(scale_factor=4, mode='nearest')
        self.shu=nn.PixelUnshuffle(1)

    def getLength(self):
        # end_params_int = torch.clamp(self.end_params, 1.0, self.block_size[0])
        # shutter = shutils.less_than(self.time_range, end_params_int)
        # self.lengths = torch.sum(shutter, dim=0)
        return self.lengths

    def get_measurement(self, input, actions, test=False):

        sig_shot_min = 0.0001
        sig_shot_max = 0.01
        sig_read_min = 0.001 / 2.
        sig_read_max = 0.03 / 2.
        # sig_shot_min = 0.0
        # sig_shot_max = 0.0
        # sig_read_min = 0.0
        # sig_read_max = 0.0
        if test:  # keep noise levels fixed when testing
            sig_shot = (sig_shot_max - sig_shot_min) / 2
            sig_read = (sig_read_max - sig_read_min) / 2
        else:
            sig_shot = (sig_shot_min - sig_shot_max) * torch.rand(1, dtype=torch.float32, device=device) + sig_shot_max
            sig_read = (sig_read_min - sig_read_max) * torch.rand(1, dtype=torch.float32, device=device) + sig_read_max

        def measure1(input):
            a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
            A = torch.from_numpy(a).to(device).reshape(1, 16)
            input = input.permute([0, 2, 3, 1]).unsqueeze(4)

            raw_measurement = torch.matmul(A.reshape(1, 1, 1, 1, 16), input).squeeze(4)
            raw_measurement = raw_measurement.permute(0, 3, 1, 2)
            raw_measurement = raw_measurement / 16.
            shoot_noise_raw_measurement = (raw_measurement.detach() ** (1 / 2)) * sig_shot * torch.randn_like(
                raw_measurement)  # B 8 H W
            read_noise_raw_measurement = sig_read * torch.randn_like(raw_measurement)

            raw_measurement = (raw_measurement + shoot_noise_raw_measurement + read_noise_raw_measurement) * 16

            raw_measurement_out = raw_measurement

            raw_measurement_out_zeros = torch.zeros_like(raw_measurement).repeat(1, 7, 1, 1)

            raw_measurement_out = torch.concat([raw_measurement_out, raw_measurement_out_zeros], dim=1)

            raw_measurement = raw_measurement.permute([0, 2, 3, 1]).unsqueeze(4)

            A_transpose = A.permute(1, 0)
            A_A_transpose = torch.matmul(A, A_transpose)
            A_A_transpose_inv = torch.linalg.inv(A_A_transpose)
            A_final = torch.matmul(A_transpose, A_A_transpose_inv)
            output = torch.matmul(A_final.reshape(1, 1, 1, 16, 1), raw_measurement).squeeze(4)
            output = output.permute(0, 3, 1, 2)
            #   z1=torch.zeros_like(output)
            #   z2=torch.zeros_like(output)
            return torch.concat([output, raw_measurement_out], dim=1)

        def measure2(input):
            a = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float32)
            A = torch.from_numpy(a).to(device).reshape(2, 16)
            input = input.permute([0, 2, 3, 1]).unsqueeze(4)

            raw_measurement = torch.matmul(A.reshape(1, 1, 1, 2, 16), input).squeeze(4)
            raw_measurement = raw_measurement.permute(0, 3, 1, 2)
            raw_measurement = raw_measurement / 16.
            shoot_noise_raw_measurement = (raw_measurement.detach() ** (1 / 2)) * sig_shot * torch.randn_like(
                raw_measurement)  # B 8 H W
            read_noise_raw_measurement = sig_read * torch.randn_like(raw_measurement)

            raw_measurement = (raw_measurement + shoot_noise_raw_measurement + read_noise_raw_measurement) * 16

            raw_measurement_out = raw_measurement

            raw_measurement_out_zeros = torch.zeros_like(raw_measurement).repeat(1, 3, 1, 1)

            raw_measurement_out = torch.concat([raw_measurement_out, raw_measurement_out_zeros], dim=1)

            # raw_measurement_out = raw_measurement.repeat(1, 8, 1, 1)

            raw_measurement = raw_measurement.permute([0, 2, 3, 1]).unsqueeze(4)

            A_transpose = A.permute(1, 0)
            A_A_transpose = torch.matmul(A, A_transpose)
            A_A_transpose_inv = torch.linalg.inv(A_A_transpose)
            # A_A_transpose_inv_=A_A_transpose_inv.detach().cpu().numpy()
            A_final = torch.matmul(A_transpose, A_A_transpose_inv)
            # A_final_ = A_final.detach().cpu().numpy()
            output = torch.matmul(A_final.reshape(1, 1, 1, 16, 2), raw_measurement).squeeze(4)
            output = output.permute(0, 3, 1, 2)

            #  raw_measurement1=torch.reshape(raw_measurement,[-1,self.block_size[1],self.block_size[2],1,2,1])
            #  raw_measurement1=torch.sum(raw_measurement1,dim=4)
            #  a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
            #  A = torch.from_numpy(a).to(device).reshape(1,16)
            #  A_transpose=A.permute(1,0)
            #  A_A_transpose=torch.matmul(A,A_transpose)
            #  A_A_transpose_inv=torch.linalg.inv(A_A_transpose)
            #  A_final=torch.matmul(A_transpose, A_A_transpose_inv)
            #  output1=torch.matmul(A_final.reshape(1,1,1,16,1),raw_measurement1).squeeze(4)
            #  output1=output1.permute(0,3,1,2)
            #  z2=torch.zeros_like(output)

            return torch.concat([output, raw_measurement_out], dim=1)

        def measure3(input):
            a = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], ], dtype=np.float32)
            A = torch.from_numpy(a).to(device).reshape(4, 16)
            input = input.permute([0, 2, 3, 1]).unsqueeze(4)

            raw_measurement = torch.matmul(A.reshape(1, 1, 1, 4, 16), input).squeeze(4)
            raw_measurement = raw_measurement.permute(0, 3, 1, 2)
            raw_measurement = raw_measurement / 16.
            shoot_noise_raw_measurement = (raw_measurement.detach() ** (1 / 2)) * sig_shot * torch.randn_like(
                raw_measurement)  # B 8 H W
            read_noise_raw_measurement = sig_read * torch.randn_like(raw_measurement)

            raw_measurement = (raw_measurement + shoot_noise_raw_measurement + read_noise_raw_measurement) * 16

            raw_measurement_out = raw_measurement

            raw_measurement_out_zeros = torch.zeros_like(raw_measurement)

            raw_measurement_out = torch.concat([raw_measurement_out, raw_measurement_out_zeros], dim=1)

            # raw_measurement_out = raw_measurement.repeat(1, 4, 1, 1)

            raw_measurement = raw_measurement.permute([0, 2, 3, 1]).unsqueeze(4)

            raw_measurement1 = torch.reshape(raw_measurement, [-1, self.block_size[1], self.block_size[2], 2, 2, 1])
            raw_measurement1 = torch.sum(raw_measurement1, dim=4)

            raw_measurement2 = torch.reshape(raw_measurement1, [-1, self.block_size[1], self.block_size[2], 1, 2, 1])
            raw_measurement2 = torch.sum(raw_measurement2, dim=4)

            A_transpose = A.permute(1, 0)
            A_A_transpose = torch.matmul(A, A_transpose)
            A_A_transpose_inv = torch.linalg.inv(A_A_transpose)
            # A_A_transpose_inv_=A_A_transpose_inv.detach().cpu().numpy()
            A_final = torch.matmul(A_transpose, A_A_transpose_inv)
            # A_final_ = A_final.detach().cpu().numpy()
            output = torch.matmul(A_final.reshape(1, 1, 1, 16, 4), raw_measurement).squeeze(4)
            output = output.permute(0, 3, 1, 2)

            #   a = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            #                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float32)
            #   A = torch.from_numpy(a).to(device).reshape(2,16)
            #   A_transpose=A.permute(1,0)
            #   A_A_transpose=torch.matmul(A,A_transpose)
            #   A_A_transpose_inv=torch.linalg.inv(A_A_transpose)
            # A_A_transpose_inv_=A_A_transpose_inv.detach().cpu().numpy()
            #   A_final=torch.matmul(A_transpose, A_A_transpose_inv)
            # A_final_ = A_final.detach().cpu().numpy()
            #
            #   output1=torch.matmul(A_final.reshape(1,1,1,16,2),raw_measurement1).squeeze(4)
            #   output1=output1.permute(0,3,1,2)

            #   a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
            #   A = torch.from_numpy(a).to(device).reshape(1,16)
            #   A_transpose=A.permute(1,0)
            #   A_A_transpose=torch.matmul(A,A_transpose)
            #  A_A_transpose_inv=torch.linalg.inv(A_A_transpose)
            #  A_final=torch.matmul(A_transpose, A_A_transpose_inv)
            #  output2=torch.matmul(A_final.reshape(1,1,1,16,1),raw_measurement2).squeeze(4)
            #  output2=output2.permute(0,3,1,2)

            return torch.concat([output, raw_measurement_out], dim=1)

        def measure4(input):
            a = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], ], dtype=np.float32)
            A = torch.from_numpy(a).to(device).reshape(8, 16)
            input = input.permute([0, 2, 3, 1]).unsqueeze(4)

            raw_measurement = torch.matmul(A.reshape(1, 1, 1, 8, 16), input).squeeze(4)
            raw_measurement = raw_measurement.permute(0, 3, 1, 2)
            raw_measurement = raw_measurement / 16.
            shoot_noise_raw_measurement = (raw_measurement.detach() ** (1 / 2)) * sig_shot * torch.randn_like(
                raw_measurement)  # B 8 H W
            read_noise_raw_measurement = sig_read * torch.randn_like(raw_measurement)

            raw_measurement = (raw_measurement + shoot_noise_raw_measurement + read_noise_raw_measurement) * 16

            raw_measurement_out = raw_measurement

            raw_measurement = raw_measurement.permute([0, 2, 3, 1]).unsqueeze(4)

            A_transpose = A.permute(1, 0)
            A_A_transpose = torch.matmul(A, A_transpose)
            A_A_transpose_inv = torch.linalg.inv(A_A_transpose)
            # A_A_transpose_inv_=A_A_transpose_inv.detach().cpu().numpy()
            A_final = torch.matmul(A_transpose, A_A_transpose_inv)
            # A_final_ = A_final.detach().cpu().numpy()
            output = torch.matmul(A_final.reshape(1, 1, 1, 16, 8), raw_measurement).squeeze(4)
            output = output.permute(0, 3, 1, 2)
            return torch.concat([output, raw_measurement_out], dim=1)

        y1 = torch.zeros_like(input)
        y2 = torch.zeros_like(input[:, :8, :, :])
        y = torch.concat([y1, y2], dim=1)

        y = torch.where(actions == 0, measure1(input), y)
        y = torch.where(actions == 1, measure2(input), y)
        y = torch.where(actions == 2, measure3(input), y)
        y = torch.where(actions == 3, measure4(input), y)

        return y




    def forward(self, video_block, train=False):
        if train:
            self.total_steps += 1
        [block_current, ref] = video_block
        

        b,c,h,w=ref.size()
        Policy_dis=self.model(self.rand_init_new).repeat(b,1,1,1)

        action = Policy_dis.argmax(dim=1)
 
   #     action=4.0*torch.ones_like(action)


        
        action=action.unsqueeze(1)

        action=action-1
    
     #    action=torch.clamp(torch.round(self.rand_init),-1,3)+(self.rand_init-self.rand_init.detach())
        
        measurement = self.get_measurement(block_current, action.detach(), train)

        # useless these two lines
        shutter=F.relu(action.detach()+1 - self.time_range + 0.9).sign()
        self.lengths = torch.sum(shutter, dim=1, keepdim=True)


        return measurement, action.detach(), None