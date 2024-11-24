import os
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
import wandb
import minari
from model import BoTModel
import matplotlib.pyplot as plt
import seaborn as sns


def average_attention_map(attention_maps, batch_average=True):

    stacked_maps = torch.stack(attention_maps)

    avg_map = stacked_maps.mean(dim=0).mean(dim=1)

    if batch_average:
        avg_map = avg_map.mean(dim=0)

    return avg_map


def visualize_attention_map(attention_map, epoch, logger=None, show_plot=False):
    attention_map = attention_map.cpu().detach().numpy()

    # Heatmap 생성
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_map,
        cmap="viridis",
        cbar=True,
        xticklabels=False,
        yticklabels=False
    )
    plt.title(f"Averaged Attention Map (Epoch: {epoch})")

    if logger is not None:
        logger.log({f"attention_map/average": wandb.Image(plt)}, step=epoch)

    if show_plot:
        plt.show()

    # 플롯 닫기
    plt.close()


def run_policy(net, env, device='cpu', mean_inputs=None, std_inputs=None):
    net.eval()

    obs, _ = env.reset()
    n_cameras = 1

    returns, frames = [], []
    success, ep, ret = 0, 0, 0

    env.unwrapped.render_mode = 'rgb_array'

    while ep < 20:
        obs = torch.from_numpy(obs).float().to(device)

        if mean_inputs is not None and std_inputs is not None:
            obs = (obs - mean_inputs.reshape(-1)) / (std_inputs.reshape(-1) + 1e-8)

        if ep == 0:
            frames.append(np.concatenate([env.render() for i in range(n_cameras)], axis=1))  # [H, W, C]

        action = net(obs.unsqueeze(0))[0].squeeze().detach().cpu().numpy()
        action = np.clip(action, -1, 1)

        next_obs, reward, terminated, truncated, info = env.step(action)
        ret += reward

        if terminated or truncated:
            success += info['success']
            obs, _ = env.reset()
            print(f"Episode {ep} return: {ret}")
            returns.append(ret)
            ret = 0
            ep += 1
        else:
            obs = next_obs

    frames = np.array(frames).transpose(0, 3, 1, 2)  # [T, C, H, W]
    success_rate = success / 20

    print(f"Mean return: {np.mean(returns):.2f}, Std return: {np.std(returns):.2f}, Success rate: {success_rate:.2f}")

    net.train()
    return np.mean(returns), np.std(returns), success_rate, frames


def main():
    wandb.init(project='BoT_Toy_Example', name="BoT-Hard")
    logger = wandb

    device = 'cuda'
    d_model, head, layer_num = 512, 4, 8
    dataset = minari.load_dataset("door-expert-v2", download=True)
    env = dataset.recover_environment()

    N = 1
    sample = dataset.sample_episodes(n_episodes=1)
    T, obs_dim = sample[0].observations.shape[0] - 1, sample[0].observations.shape[-1]
    action_dim = sample[0].actions.shape[-1]
    print(f"N: {N}, T: {T}, obs_dim: {obs_dim}, action_dim: {action_dim}")

    inputs = torch.zeros((N, T, obs_dim))
    targets = torch.zeros((N, T, action_dim))

    dataset.set_seed(42)
    episodes = dataset.sample_episodes(n_episodes=N)
    for i, episode in enumerate(episodes):
        inputs[i] = torch.from_numpy(episode.observations[:-1])
        targets[i] = torch.from_numpy(episode.actions)

    new_inputs = inputs.view(N * T, obs_dim)
    targets = targets.view(N * T, action_dim)

    mean_inputs, std_inputs = new_inputs.mean(dim=0), new_inputs.std(dim=0)

    new_inputs = (new_inputs - mean_inputs) / (std_inputs + 1e-8)

    train_dataset = torch.utils.data.TensorDataset(new_inputs, targets)
    print(f"Train dataset size: {len(train_dataset)}")

    model = BoTModel(d_model=d_model, head=head, layer_num=layer_num, mode='Hard').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

    mean_inputs = mean_inputs.to(device)
    std_inputs = std_inputs.to(device)

    for epoch in range(100):
        total_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, attention_maps = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.3f}")
        logger.log({'loss': avg_loss}, step=epoch)

        avg_attention_map = average_attention_map(attention_maps)
        visualize_attention_map(avg_attention_map, epoch, logger=logger)

        mean, std, success_rate, frames = run_policy(model, env, device=device, mean_inputs=mean_inputs, std_inputs=std_inputs)
        logger.log({'mean_return': mean, 'std_return': std, 'success_rate': success_rate}, step=epoch)
        logger.log({'video/video': wandb.Video(frames, fps=30, format='mp4')}, step=epoch)


if __name__ == '__main__':
    main()
