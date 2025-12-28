import torch
import tqdm as TQDM
import torch.optim as optim

from mftv.losses import combined_loss, backward_loss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


class Trainer:
    def __init__(self, model, source_dist, target_dist, batch_size, method):
        self.model = model.to(device)  # move once
        self.source = source_dist
        self.target = target_dist
        self.batch_size = batch_size
        self.method = method

    def sample_t_and_r(self, n):
        s = torch.rand(n, 2, device=device)
        # Assign the smaller values to r, larger values to t, unsqueeze to make it fit the 2D data
        t = torch.max(s[:, 0], s[:, 1]).unsqueeze(1)
        r = torch.min(s[:, 0], s[:, 1]).unsqueeze(1)
        return t, r

    def calc_loss_batch(self):
        # generate on CPU then move once to device (your generator may already provide tensors)
        src = self.source.generate(self.batch_size).to(device)
        tgt = self.target.generate(self.batch_size).to(device)
        t, r = self.sample_t_and_r(self.batch_size)

        if self.method == 'Original':
            return backward_loss(self.model, src, tgt, t, r)
        else:
            return combined_loss(self.model, src, tgt, t, r)



    def train_mean_flow_model(self, num_batches_per_epoch, n_epochs,
                            lr):
        opt = optim.AdamW(self.model.parameters(), lr=lr)

        pbar = TQDM.tqdm(range(n_epochs), desc=f"Training {self.method}")

        # Initialize tracking dictionaries

        for epoch in pbar:
            epoch_loss = 0.0
            # Batch metrics for this epoch
            for batch_idx in range(num_batches_per_epoch):
                loss = self.calc_loss_batch()
                opt.zero_grad()
                loss.backward()


                opt.step()
                # scheduler.step(loss)

                epoch_loss += loss.item()


            # Store average metrics for this epoch
            avg_loss = epoch_loss / num_batches_per_epoch


            pbar.set_description(f"{self.method} Formulation Epoch {epoch} Loss {avg_loss:.4f} ")

        return self.model.to('cpu')
