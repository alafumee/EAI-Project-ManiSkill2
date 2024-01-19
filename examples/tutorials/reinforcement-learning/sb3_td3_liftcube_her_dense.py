# Import required packages
import argparse
from collections import OrderedDict
import os.path as osp
from typing import Any

import gymnasium as gym
from gymnasium.core import Env
import numpy as np
from stable_baselines3 import PPO, SAC, TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

import mani_skill2.envs
from mani_skill2.utils.common import convert_observation_to_space, flatten_state_dict
from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3.common.envs import BitFlippingEnv
from td3_her import TD3_HER



# Defines a continuous, infinite horizon, task where terminated is always False
# unless a timelimit is reached.
class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)

    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        return ob, rew, False, truncated, info


# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
    
    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, terminated, truncated, info


def format_observation(ob):
    new_ob = OrderedDict(
            observation=flatten_state_dict(ob)
        )
    obj_goal_height = ob["extra"]["goal_height"]
    obj_height = ob["extra"]["obj_height"]
    robot_qvel = ob["agent"]["qvel"][:-2]
    goal_qvel = np.zeros_like(robot_qvel)
    desired_goal = np.hstack([obj_goal_height, goal_qvel])
    achieved_goal = np.hstack([obj_height, robot_qvel])
    new_ob.update(desired_goal=desired_goal, achieved_goal=achieved_goal)
    return new_ob


class LiftCubeHERWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        ob, rew, terminated, truncated, info = super().step(action)
        assert isinstance(ob, dict)
        ob["extra"].update(
            goal_height=self.env.goal_pos[2],
            obj_height=self.env.obj.pose.p[2],
        )
        new_ob = format_observation(ob)
        is_grasped = self.env.agent.check_grasp(self.env.obj, max_angle=30)
        tcp_pos = self.env.tcp.pose.p
        info.update(
            is_grasped=is_grasped,
            real_goal=new_ob["desired_goal"],
            tcp_pos = tcp_pos,
            obj_pos_3d = self.env.obj.pose.p,
        )
        return new_ob, rew, terminated, truncated, info
    
    def reset(self, *args, **kwargs):
        ob, info = super().reset(*args, **kwargs)
        ob["extra"].update(
            goal_height=self.env.goal_pos[2],
            obj_height=self.env.obj.pose.p[2],
        )
        new_ob = format_observation(ob)
        return new_ob, info
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        # fn = lambda x: x["success"]
        # assert np.array(list(map(fn, info))).any() == False
        thresh = 0.2
        EPS = 1e-4
        # if len(achieved_goal.shape) == 1:
        #     batch_size = 1
        #     goal_pos = desired_goal[:3]
        #     obj_pos = achieved_goal[:3]
        #     robot_qvel = achieved_goal[3:]
        #     is_obj_placed = np.linalg.norm(goal_pos - obj_pos) <= self.env.goal_thresh
        #     is_robot_static = np.max(np.abs(robot_qvel)) <= thresh
        #     success = float(is_obj_placed and is_robot_static) 
        # else:
        assert len(achieved_goal.shape) == 2
        if achieved_goal.shape[0] == 0:
            return np.array([], dtype=int)
        batch_size = achieved_goal.shape[0]
        goal_height = desired_goal[:, :1]
        obj_height = achieved_goal[:, :1]
        robot_qvel = achieved_goal[:, 1:]
        is_obj_placed = (obj_height >= goal_height).squeeze()
        is_robot_static = np.max(np.abs(robot_qvel), axis=-1) <= thresh
        success = np.logical_and(is_obj_placed, is_robot_static).astype(np.float32)
        
        # orig_rew = np.array(list(map(lambda x: x["reward"], info)))
        # orig_success = np.array(list(map(lambda x: x["success"], info)))
        # is_grasped = np.array(list(map(lambda x: x["is_grasped"] if "is_grasped" in x else None, info)))
        # place_reward = (1 - np.tanh(5 * np.linalg.norm(goal_pos - obj_pos, axis=-1)))
        # real_goal_pos = np.array(list(map(lambda x: x["real_goal"][:3], info)))
        # orig_place_reward = (1 - np.tanh(5 * np.linalg.norm(real_goal_pos - obj_pos, axis=-1)))
        # rew = 5 * success + (orig_rew + (place_reward - orig_place_reward) * is_grasped) * (1 - success)
        
        tcp_pos = np.array(list(map(lambda x: x["tcp_pos"], info)))
        obj_pos_3d = np.array(list(map(lambda x: x["obj_pos_3d"], info)))
        is_grasped = np.array(list(map(lambda x: x["is_grasped"] if "is_grasped" in x else None, info)))
        
        reaching_reward = (1 - np.tanh(5 * np.linalg.norm(obj_pos_3d - tcp_pos, axis=-1)))
        lift_reward = np.minimum(1.0, (obj_height - self.env.cube_half_size[2] + EPS) / (goal_height - self.env.cube_half_size[2] + EPS)).squeeze()
        
        rew = 2.25 * success + (reaching_reward + (lift_reward + 0.25)* is_grasped) * (1 - success)
        return rew / 2.25
    
    def update_observation_space(self):
        obs, _ = self.reset(seed=2023)
        self.env.observation_space = convert_observation_to_space(obs)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple script demonstrating how to use Stable Baselines 3 with ManiSkill2 and RGBD Observations"
    )
    parser.add_argument("-e", "--env-id", type=str, default="LiftCube-v0")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=8,
        help="number of parallel envs to run. Note that increasing this does not increase rollout size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to initialize training with",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=50,
        help="Max steps per episode before truncating them",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="path for where logs, checkpoints, and videos are saved",
    )
    parser.add_argument(
        "--eval", action="store_true", help="whether to only evaluate policy"
    )
    parser.add_argument(
        "--model-path", type=str, help="path to sb3 model for evaluation"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    env_id = args.env_id
    num_envs = args.n_envs
    max_episode_steps = args.max_episode_steps
    log_dir = args.log_dir
    rollout_steps = 4800

    obs_mode = "state_dict"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "normalized_dense"
    if args.seed is not None:
        set_random_seed(args.seed)

    def make_env(
        env_id: str,
        max_episode_steps: int = None,
        record_dir: str = None,
    ):
        def _init() -> gym.Env:
            # NOTE: Import envs here so that they are registered with gym in subprocesses
            import mani_skill2.envs

            env = gym.make(
                env_id,
                obs_mode=obs_mode,
                reward_mode=reward_mode,
                control_mode=control_mode,
                render_mode="cameras",
                max_episode_steps=max_episode_steps,
            )
            # For training, we regard the task as a continuous task with infinite horizon.
            # you can use the ContinuousTaskWrapper here for that
            env = LiftCubeHERWrapper(env)
            env.update_observation_space()
            if max_episode_steps is not None:
                env = ContinuousTaskWrapper(env)
            if record_dir is not None:
                env = SuccessInfoWrapper(env)
                env = RecordEpisode(env, record_dir, info_on_video=True)
            return env

        return _init

    # create eval environment
    if args.eval:
        record_dir = osp.join(log_dir, "videos/eval")
    else:
        record_dir = osp.join(log_dir, "videos")
    eval_env = SubprocVecEnv(
        [make_env(env_id, record_dir=record_dir) for _ in range(1)]
    )
    eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
    eval_env.seed(args.seed)
    eval_env.reset()

    if args.eval:
        env = eval_env
    else:
        # Create vectorized environments for training
        env = SubprocVecEnv(
            [
                make_env(env_id, max_episode_steps=max_episode_steps)
                for _ in range(num_envs)
            ]
        )
        env = VecMonitor(env)
        env.seed(args.seed)
        env.reset()

    # Define the policy configuration and algorithm configuration
    policy_kwargs = dict(net_arch=[256, 256])
    her_num_fn = lambda x: 0 # lambda x: 0.5 if x >= 250000 else 0.0
    goal_selection_strategy = "future"
    model = TD3_HER(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        batch_size=400,
        gamma=0.8,
        tensorboard_log=log_dir,
        train_freq=50,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=0.25,
            goal_selection_strategy=goal_selection_strategy,
            copy_info_dict=True,
        ),
        learning_starts=100,
        her_num_fn=her_num_fn,
    )

    if args.eval:
        model_path = args.model_path
        if model_path is None:
            model_path = osp.join(log_dir, "latest_model")
        # Load the saved model
        model = model.load(model_path, env=env)
    else:
        # define callbacks to periodically save our model and evaluate it to help monitor training
        # the below freq values will save every 10 rollouts
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=10 * rollout_steps // num_envs,
            deterministic=True,
            render=False,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=10 * rollout_steps // num_envs,
            save_path=log_dir,
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        # Train an agent with PPO for args.total_timesteps interactions
        model.learn(
            args.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
        )
        # Save the final model
        model.save(osp.join(log_dir, "latest_model"))

    # Evaluate the model
    returns, ep_lens = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
        n_eval_episodes=50,
    )
    print("Returns", returns)
    print("Episode Lengths", ep_lens)
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print("Success Rate:", success_rate)

    # close all envs
    eval_env.close()
    if not args.eval:
        env.close()


if __name__ == "__main__":
    main()
