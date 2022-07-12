'''
python test_learned_prior.py \
--model_path logs/fp_epoch100_2_cycle/rnn_size\=256-predictor-posterior-rnn_layers\=2-1-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000/model.pth \
--log_dir logs/fp_epoch100_2_cycle/rnn_size\=256-predictor-posterior-rnn_layers\=2-1-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000/

python test_fixed_prior.py \
--model_path logs/fp_epoch100_no_cycle/rnn_size\=256-predictor-posterior-rnn_layers\=2-1-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000/model.pth \
--log_dir logs/fp_epoch100_no_cycle/rnn_size\=256-predictor-posterior-rnn_layers\=2-1-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000/

'''
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
from tqdm import tqdm
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from dataset import bair_robot_pushing_dataset
from models.lstm import gaussian_lstm, lstm
from models.vgg_64 import vgg_decoder, vgg_encoder
from utils import finn_eval_seq, pred_lp


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--data_root', default='data/processed_data', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=1, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=3, help='number of samples')
parser.add_argument('--N', type=int, default=3, help='number of samples')


args = parser.parse_args()
os.makedirs('%s' % args.log_dir, exist_ok=True)


args.n_eval = args.n_past+args.n_future
args.max_step = args.n_past + args.n_future

print("Random Seed: ", args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dtype = torch.cuda.FloatTensor



# ---------------- load the models  ----------------
modules = torch.load(args.model_path)
frame_predictor = modules['frame_predictor']
posterior = modules['posterior']
prior = modules['prior']
frame_predictor.eval()
posterior.eval()
prior.eval()
encoder = modules['encoder']
decoder = modules['decoder']
encoder.eval()
decoder.eval()
frame_predictor.batch_size = args.batch_size
posterior.batch_size = args.batch_size
prior.batch_size = args.batch_size
args.g_dim = modules['args'].g_dim
args.z_dim = modules['args'].z_dim

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()

# ---------------- set the argsions ----------------
args.last_frame_skip = modules['args'].last_frame_skip

print(args)


# --------- load a dataset ------------------------------------
test_data = bair_robot_pushing_dataset(args, 'test')

test_loader = DataLoader(test_data,
                         num_workers=args.num_threads,
                         batch_size=args.batch_size,
                         shuffle=False,
                         drop_last=True,
                         pin_memory=True)

test_iterator = iter(test_loader)


# --------- eval funtions ------------------------------------

def make_gifs(x, cond, idx, name):
    # get approx posterior sample
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    x_in = x[0]
    for i in range(1, args.n_eval):
        h = encoder(x_in)
        h_target = encoder(x[i])[0].detach()
        if args.last_frame_skip or i < args.n_past:	
            h, skip = h
        else:
            h, _ = h
        h = h.detach()
        _, z_t, _= posterior(h_target) # take the mean
        if i < args.n_past:
            frame_predictor(torch.cat([cond[i-1], h, z_t], 1)) 
            posterior_gen.append(x[i])
            x_in = x[i]
        else:
            h_pred = frame_predictor(torch.cat([cond[i-1], h, z_t], 1)).detach()
            x_in = decoder([h_pred, skip]).detach()
            posterior_gen.append(x_in)
  

    nsample = args.nsample
    ssim = np.zeros((args.batch_size, nsample, args.n_future))
    psnr = np.zeros((args.batch_size, nsample, args.n_future))
    progress = tqdm(total=nsample)
    all_gen = []
    for s in range(nsample):
        progress.update(1)
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, args.n_eval):
            h = encoder(x_in)
            if args.last_frame_skip or i < args.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()

            if i < args.n_past:
                h_target = encoder(x[i])[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([cond[i-1], h, z_t], 1))
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([cond[i-1], h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in)
                gt_seq.append(x[i])
                all_gen[s].append(x_in)
        # print(len(gt_seq[args.n_past:]))
        _, ssim[:, s, :], psnr[:, s, :] = finn_eval_seq(gt_seq, gen_seq)


    ###### ssim ######
    for i in range(args.batch_size):
        gifs = [ [] for t in range(args.n_eval) ]
        text = [ [] for t in range(args.n_eval) ]
        mean_psnr = np.mean(psnr[i], 1)
        ordered = np.argsort(mean_psnr)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for t in range(args.n_eval):
            # gt 
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append('Ground\ntruth')
            #posterior 
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')
            # best 
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best PSNR')
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))

        fname = '%s/%s_%d.gif' % (args.log_dir, name, idx+i) 
        utils.save_gif_with_text(fname, gifs, text)

def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px

if __name__ == '__main__':
    # plot test
    device = 'cuda'
    psnr_list = []
    for i, (test_seq, test_cond) in enumerate(tqdm(test_loader)):
        test_seq = test_seq.permute(1, 0, 2, 3 ,4).to(device)
        test_cond = test_cond.permute(1, 0, 2).to(device)
        pred_seq = pred_lp(test_seq, test_cond, modules, args, device)
        _, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past:])
        psnr_list.append(psnr)
    ave_psnr = np.mean(np.concatenate(psnr_list))
    print(f'====================== test psnr = {ave_psnr:.5f} ========================')

    test_iterator = iter(test_loader)
    test_seq, test_cond = next(test_iterator)
    test_seq = test_seq.permute(1, 0, 2, 3 ,4).to(device)
    test_cond = test_cond.permute(1, 0, 2).to(device)
    make_gifs(test_seq, test_cond, 0, 'test')
