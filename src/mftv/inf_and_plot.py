import os

import torch
import matplotlib.pyplot as plt

from mftv.distribution import GaussianGenerator, CrescentGenerator


def forward_euler_simulator(model, source_samples, n_steps=50):

    device = next(model.parameters()).device
    xs = source_samples.to(device)
    B, d = xs.shape

    positions = torch.zeros(n_steps, B, d, device=device)
    positions[0] = xs

    t_space = torch.linspace(0.0, 1.0, n_steps, device=device)

    for i in range(1, n_steps):
        cur = positions[i-1]                       # <-- FIXED
        cur_t = t_space[i].unsqueeze(0)            # t = t_i
        cur_r = t_space[i-1].unsqueeze(0)          # r = t_{i-1}


        vel = model(cur,
                    cur_t.expand(B, 1),
                    cur_r.expand(B, 1))

        dt = t_space[i] - t_space[i-1]
        positions[i] = cur - dt * vel

    return t_space.cpu(), positions.cpu()


def plot_and_save_trajectories(models, n_samples=1000, n_steps=100, save_dir="trajectory_plots"):
    os.makedirs(save_dir, exist_ok=True)
    p_init = GaussianGenerator(n_dims=2, noise_std=0.5)
    p_target = CrescentGenerator(R=1.0, r=0.6, d=0.5)

    src = p_init.generate(n_samples)
    endpoints = {}
    plt.figure(figsize=(18, 6))

    colors = {'Original': 'red', 'Combined': 'blue'}

    for i, (method, model) in enumerate(models.items()):
        model_cpu = model.to('cpu')
        t_space, positions = forward_euler_simulator(model_cpu, src, n_steps=n_steps)
        endpoints[method] = positions[-1]
        plt.subplot(1, 3, i+1)

        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.gca().set_aspect('equal')



        # Batch scatter for efficiency
        end_points = positions[-1].detach().numpy()
        plt.scatter(end_points[:, 0], end_points[:, 1],
                   s=20, c=colors[method], marker='s', label='End', alpha=0.6)


        plt.title(f"Model Trained with {method} Loss")
        plt.legend(loc='upper right')

    plt.subplot(1, 3, 3)
    crescent_samples = p_target.generate(n_samples)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.gca().set_aspect('equal')
    plt.scatter(crescent_samples[:, 0], crescent_samples[:, 1], c='green', alpha=0.7, s=8, label='Target Crescent')
    plt.title("Target Distribution")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/crescent_comparison.pdf", dpi=200)
    plt.close()
