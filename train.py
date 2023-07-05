import os
import time
import inspect

from termcolor import colored, cprint
from tqdm import tqdm

# profiler
from torch import profiler

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from options.train_options import TrainOptions
from datasets.dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model

from utils.visualizer import Visualizer
from utils import util
import numpy as np

opt = TrainOptions().parse()
opt.phase = 'train'

seed = opt.seed
util.seed_everything(seed)

train_dl, test_dl = CreateDataLoader(opt)
train_ds, test_ds = train_dl.dataset, test_dl.dataset

test_dg = get_data_generator(test_dl)

dataset_size = len(train_ds)
if opt.dataset_mode == 'shapenet_lang':
    cprint('[*] # training text snippets = %d' % len(train_ds), 'yellow')
    cprint('[*] # testing text snippets = %d' % len(test_ds), 'yellow')
else:
    cprint('[*] # training images = %d' % len(train_ds), 'yellow')
    cprint('[*] # testing images = %d' % len(test_ds), 'yellow')

# main loop
model = create_model(opt)
cprint(f'[*] "{opt.model}" initialized.', 'cyan')

visualizer = Visualizer(opt)

# save model and dataset files
save_loss = os.path.join(opt.logs_dir, opt.name)
expr_dir = '%s/%s' % (opt.logs_dir, opt.name)
model_f = inspect.getfile(model.__class__)
dset_f = inspect.getfile(train_ds.__class__)
cprint(f'[*] saving model and dataset files: {model_f}, {dset_f}', 'blue')
modelf_out = os.path.join(expr_dir, os.path.basename(model_f))
dsetf_out = os.path.join(expr_dir, os.path.basename(dset_f))
os.system(f'cp {model_f} {modelf_out}')
os.system(f'cp {dset_f} {dsetf_out}')

if opt.vq_cfg is not None:
    vq_cfg = opt.vq_cfg
    cfg_out = os.path.join(expr_dir, os.path.basename(vq_cfg))
    os.system(f'cp {vq_cfg} {cfg_out}')
    
if opt.tf_cfg is not None:
    tf_cfg = opt.tf_cfg
    cfg_out = os.path.join(expr_dir, os.path.basename(tf_cfg))
    os.system(f'cp {tf_cfg} {cfg_out}')


# use profiler or not
if opt.profiler == '1':
    cprint("[*] Using pytorch's profiler...", 'blue')
    tensorboard_trace_handler = profiler.tensorboard_trace_handler(opt.tb_dir)
    schedule_args = {'wait': 2, 'warmup': 2, 'active': 6, 'repeat': 1}
    schedule = profiler.schedule(**schedule_args)
    activities = [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA]


print(len(train_ds),"Length train dataset")
print(len(test_ds),"Length test dataset")
print(len(train_dl),"Length train_dl")
print(len(test_dl), "Length test_dl")
################## main training loops #####################
def train_one_epoch(pt_profiler=None):
    global total_steps
    global train_loss
    global valid_loss
    global save_loss
    
    epoch_iter = 0
    for i, data in tqdm(enumerate(train_dl), total=len(train_dl)):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        model.set_input(data)

        model.optimize_parameters(total_steps)

        nBatches_has_trained = total_steps // opt.batch_size

        # if total_steps % opt.print_freq == 0:
        if nBatches_has_trained % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_errors(epoch, epoch_iter, total_steps, errors, t)

        if (nBatches_has_trained % opt.display_freq == 0) or i == 0:
            # eval
            t = (time.time() - iter_start_time) / opt.batch_size
            model.inference(data)
            errors = model.get_current_errors()
            visualizer.print_current_errors(epoch, epoch_iter, total_steps, errors, t)
            train_loss.append({"step":total_steps,"loss":errors["nll"].item()})
            k = np.array(train_loss)
            np.save(f"{save_loss}/train_loss", k)
            #visualizer.display_current_results(model.get_current_visuals(), total_steps, phase='train')
            
            # model.set_input(next(test_dg))
            #import pdb;pdb.set_trace()
            test_data = next(test_dg)
            model.inference(test_data)
            errors = model.get_current_errors()
            visualizer.print_current_errors(epoch, epoch_iter, total_steps, errors, t)
            valid_loss.append({"step":total_steps,"loss":errors["nll"].item()})
            k = np.array(valid_loss)
            np.save(f"{save_loss}/valid_loss", k)
            #visualizer.display_current_results(model.get_current_visuals(), total_steps, phase='test')

        if total_steps % opt.save_latest_freq == 0:
            cprint('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps), 'blue')
            latest_name = f'epoch-latest'
            model.save(latest_name)
        
        if pt_profiler is not None:
            pt_profiler.step()


cprint('[*] Start training. name: %s' % opt.name, 'blue')
total_steps = 0
train_loss = []
valid_loss = []
#save_loss = os.path.join(opt.logs_dir, opt.name, 'losses')
for epoch in range(opt.nepochs + opt.nepochs_decay):
    epoch_start_time = time.time()
    # epoch_iter = 0

    # profile
    if opt.profiler == '1':
         with profiler.profile(
            schedule=schedule,
            activities=activities,
            on_trace_ready=tensorboard_trace_handler,
            record_shapes=True,
            with_stack=True,
        ) as pt_profiler:
            train_one_epoch(pt_profiler)
    else:
        train_one_epoch()

    if epoch % opt.save_epoch_freq == 0:
        cprint('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps), 'blue')
        latest_name = f'epoch-latest'
        model.save(latest_name)
        cur_name = f'epoch-{epoch}'
        model.save(cur_name)

    # eval every 3 epoch
    if epoch % opt.save_epoch_freq == 0:
        metrics = model.eval_metrics(test_dl)
        visualizer.print_current_metrics(epoch, metrics, phase='test')
        print(metrics)

    cprint(f'[*] End of epoch %d / %d \t Time Taken: %d sec \n%s' %
        (
            epoch, opt.nepochs + opt.nepochs_decay,
            time.time() - epoch_start_time,
            os.path.abspath( os.path.join(opt.logs_dir, opt.name) )
        ), 'blue', attrs=['bold']
        )
    model.update_learning_rate()

