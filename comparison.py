from distribution import GaussianGenerator, CrescentGenerator
from model import MeanFlowModel
from trainer import Trainer
import torch
import torch.optim as optim
import tqdm as TQDM
from inf_and_plot import plot_and_save_trajectories



if __name__ == "__main__":

    batch_size = 1024
    num_batches_per_epoch = 100  
    n_epochs = 50

    p_init = GaussianGenerator(n_dims=2, noise_std=0.5)
    p_target = CrescentGenerator(R=1.0, r=0.6, d=0.5)

    model_original = MeanFlowModel(input_dim=2, output_dim=2, dim=256, n_hidden=2)
    model_combined = MeanFlowModel(input_dim=2, output_dim=2, dim=256, n_hidden=2)

    trainer_backward = Trainer(model_combined, p_init, p_target, batch_size, method='Original')
    trainer_combined = Trainer(model_original, p_init, p_target, batch_size, method='Combined')

    model_original = trainer_backward.train_mean_flow_model( num_batches_per_epoch, n_epochs, lr=1e-3)
    
    model_combined = trainer_combined.train_mean_flow_model(num_batches_per_epoch, n_epochs, lr=1e-3)

    model_dict = { 'Original': model_original, 'Combined': model_combined }
    plot_and_save_trajectories(model_dict, n_samples=2048, n_steps=10, save_dir="trajectory_plots")
