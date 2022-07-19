import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config
import os
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmarks = True

checkpoint_dir = './train/check_points/'
output_dir = './train/output/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.IMG_CHANNELS)],
                [0.5 for _ in range(config.IMG_CHANNELS)],
            )
        ]
    )
    batch_size = config.BATCH_SIZE[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(dataset,
                        batch_size,
                        shuffle=True,
                        num_workers=config.NUM_WORKERS,
                        pin_memory=True,
                        drop_last=True)
    return loader, dataset


def train_fn(critic, gen, loader, dataset, step,
             alpha, opt_critic, opt_gen, scaler_gen, scaler_critic, iter_num, output_dir):
    loop = tqdm(loader)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        alpha += cur_batch_size / (
                (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        iter_num += 1

        if iter_num % 500 == 0:
            with torch.no_grad():
                # fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
                # fixed_fakes = make_grid(fixed_fakes, 4, 0)
                # save_image(fixed_fakes, output_dir + 'gen_{}.jpg'.format(iter_num))
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
                fixed_fakes = make_grid(fixed_fakes, nrow=4, scale_each=True,
                                        padding=int(0.5 * (2 ** (step+1)))).permute(1, 2, 0)
                plt.imshow(fixed_fakes.cpu())
                plt.savefig(output_dir + 'gen_{}'.format(iter_num))


        loop.set_postfix(gp=gp.item(),
                         loss_critic=loss_critic.item(),
                         loss_gen=loss_gen.item())
    return alpha, iter_num


def main():
    gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    critic = Discriminator(config.IN_CHANNELS, img_channels=config.IMG_CHANNELS).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    scaler_critic = torch.cuda.amp.GradScaler()
    scaler_gen = torch.cuda.amp.GradScaler()

    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
    #     )

    gen.train()
    critic.train()
    iter_num = 0
    iter_epoch = 0
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4*2**step)
        print(f"Image size: {4*2**step}")
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            alpha, iter_num = train_fn(critic, gen, loader, dataset, step,
                             alpha, opt_critic, opt_gen, scaler_gen, scaler_critic, iter_num, output_dir)

            checkpoint = {
                'G_net': gen.state_dict(),
				'G_optimizer': opt_gen.state_dict(),
				'D_net': critic.state_dict(),
				'D_optimizer': opt_critic.state_dict(),
				'depth': step,
				'alpha': alpha
				   }
            if config.SAVE_MODEL:
                torch.save(checkpoint, checkpoint_dir + 'check_point_epoch_%d.pth' % (iter_epoch+1))
            iter_epoch += 1
        step += 1


if __name__ == "__main__":
    main()
