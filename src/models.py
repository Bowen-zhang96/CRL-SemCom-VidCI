import torch
import torch.nn as nn
import shutters.shutters_adaptive5_nomask as shutters

device = 'cuda:0'
from nets.MST_Plus_Plus_nomask import MST_Plus_Plus


def define_shutter(shutter_type, args, test=False, model_dir=''):
    return shutters.Shutter(shutter_type=shutter_type, block_size=args.block_size,
                            test=test, resume=args.resume, model_dir=model_dir, init=args.init)


def define_model(shutter, decoder, args, get_coded=False):
    ''' Define any special interpolation modules in between encoder and decoder '''
    if args.shutter in ['short', 'med', 'long', 'full'] or decoder is None or args.interp is None:
        # print('***** No interpolation module added! *****')
        return Model(shutter, decoder, dec_name=args.decoder, get_coded=get_coded)



def define_decoder(model_name, args):
    if args.decoder == 'none':
        return None
    out_ch = 16
    if args.shutter == 'full':
        in_ch = 3
    elif args.shutter in ['short', 'med', 'long'] or args.interp is None:
        in_ch = 1
    elif 'quad' in args.shutter:
        in_ch = 4
    elif 'nonad' in args.shutter:
        in_ch = 9
    elif args.interp == 'scatter':
        in_ch = 9
    else:
        raise NotImplementedError

    if model_name == 'MST':
        model = MST_Plus_Plus()
        return model

    raise NotImplementedError('Model not specified correctly')



class Model(nn.Module):
    def __init__(self, shutter, decoder, dec_name=None, get_coded=False):
        super().__init__()
        self.get_coded = get_coded
        self.shutter = shutter
        self.decoder = decoder

        self.dec_name = dec_name

    def forward(self, input, train=True, steps=None):
        coded1, actions, rate_loss = self.shutter(input, train=train, steps=steps)
        if not coded1.requires_grad:
            ## needed for computing gradients wrt input for fixed shutters
            coded1.requires_grad = True

        if self.decoder is None:
            if self.get_coded:
                return coded1, coded1
            return coded1
        # self.decoder.eval()
        # x = self.decoder(torch.concat([coded1,actions],dim=1))
        x = self.decoder(coded1, actions)
        # x=coded1[:,:-1,:,:]

        if self.get_coded:
            return x, coded1
        return x, coded1, actions, rate_loss

    def forward_using_capture(self, coded):
        x = self.decoder(coded)
        if self.get_coded:
            return x, coded
        return x
