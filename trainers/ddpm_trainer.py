import copy
import wandb
import torch
import numpy as np
from pathlib import Path
from functools import partial
from torchvision import utils
from torch.optim import Adam

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class DDPM_Trainer(object):
    def __init__(
        self,
        diffusion_model,
        *,
        dataloader=None,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        fp16=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        results_folder='./results',
        config={}
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        
        self.dl = cycle(dataloader)
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.device = config['device']
        self.config = config

        self.step = 0

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        # Initialize a new wandb run
        wandb.init(project='ddpm-test', config=self.config)
        wandb.watch(self.model)

        backwards = partial(loss_backwards, self.fp16)
        while self.step < self.train_num_steps:
            train_loss = []
            for _ in range(self.gradient_accumulate_every):
                data = next(self.dl)[0].to(self.device)
                loss = self.model(data)
                # print(f'{self.step}: {loss.item()}')
                backwards(loss / self.gradient_accumulate_every, self.opt)
                train_loss.append(loss.item())

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                # compute milestone number
                milestone = self.step // self.save_and_sample_every
                
                # generate samples
                batches = num_to_groups(36, self.batch_size)
                all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5

                # log samples to wandb
                img_path = str(self.results_folder / f'sample-{milestone}-{self.config["model"]}-{self.config["dataset"]}.png')
                utils.save_image(all_images, img_path, nrow = 6)
                wandb.log({"Sample": wandb.Image(img_path)}, commit=False)
                os.remove(img_path)
                
                # save model
                # self.save(milestone)
            
            # update step
            self.step += 1
            
            # log to wandb
            train_loss = np.array(train_loss).mean()
            wandb.log({'train_loss': train_loss}, commit=True)

        print('training completed')
