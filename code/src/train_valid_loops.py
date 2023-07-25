import torch


def train(policy, env, optimizer, criterion, wandb=None, episode=None):
    policy.train()
    device = next(policy.parameters()).device
    input_tensordict = env.reset()
    done = input_tensordict["done"][0].item()
    episode_train_loss = 0.
    num_steps = 0
    # t_eval_emb, t_eval_qval, t_eval_step, t_eval_loss = 0, 0, 0, 0
    while not done:
        optimizer.zero_grad()
        # ts = time.time()
        # input_tensordict = policy[0](input_tensordict.to(device))
        # t_emb = time.time()
        # input_tensordict = policy[1](input_tensordict).to(device)
        # t_qval = time.time()
        input_tensordict = policy(input_tensordict.to(device))
        input_tensordict = env.step(input_tensordict.to(env.device))
        # t_step = time.time()
        loss = criterion(input_tensordict)["loss"]
        # t_loss = time.time()
        loss.backward()
        optimizer.step()

        if wandb is not None:
            wandb.log({"Episode": episode, "Step": num_steps, "train-batch Loss": loss.item()})
        episode_train_loss += loss.item()
        num_steps += 1
        input_tensordict = input_tensordict["next"]
        done = input_tensordict["done"][0].item()
    #         t_eval_emb += t_emb - ts
    #         t_eval_qval += t_qval - t_emb
    #         t_eval_step += t_step - t_qval
    #         t_eval_loss += t_loss - t_step
    #     t_eval_emb /= num_steps
    #     t_eval_qval /= num_steps
    #     t_eval_step /= num_steps
    #     t_eval_loss /= num_steps
    #     print(f"Time Emb: {t_eval_emb:.2f}s; Time QVal: {t_eval_qval:.2f}s; Time Step: {t_eval_step:.2f}s; Time Loss: {t_eval_loss:.2f}s;")
    return episode_train_loss / num_steps, input_tensordict["users_rewards"].float().mean(1).mean().item()


@torch.no_grad()
def valid(policy, env, criterion, n_valid_episodes=1):
    policy.eval()
    device = next(policy.parameters()).device
    valid_loss_val, valid_metric_val = 0, 0
    for episode in range(n_valid_episodes):
        input_tensordict = env.reset()
        done = input_tensordict["done"][0].item()
        episode_valid_loss = 0.
        num_steps = 0
        while not done:
            input_tensordict = policy(input_tensordict.to(device))
            input_tensordict = env.step(input_tensordict.to(env.device))
            loss = criterion(input_tensordict)["loss"]
            episode_valid_loss += loss.item()
            num_steps += 1
            input_tensordict = input_tensordict["next"]
            done = input_tensordict["done"][0].item()
        valid_loss_val += episode_valid_loss / num_steps / n_valid_episodes
        valid_metric_val += input_tensordict["users_rewards"].float().mean(1).mean().item() / n_valid_episodes
    return valid_loss_val, valid_metric_val
