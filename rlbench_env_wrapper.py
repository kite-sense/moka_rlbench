from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from rlbench.tasks import *  # 示例任务

import numpy as np


# RLBench 环境包装器
class RLBenchEnvWrapper:
    def __init__(self, task_name="close_box", headless=False):
        # 设置动作模式 - 使用笛卡尔位置控制
        self.action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete()
        )

        # 配置观察
        self.obs_config = ObservationConfig()
        self.obs_config.set_all(True)  # 启用所有观察

        self.obs_config.left_shoulder_camera.rgb = True
        self.obs_config.right_shoulder_camera.rgb = True
        self.obs_config.front_camera.rgb = True
        self.obs_config.wrist_camera.rgb = True
        self.obs_config.overhead_camera.rgb = True

        # 启用点云
        self.obs_config.left_shoulder_camera.point_cloud = True
        self.obs_config.right_shoulder_camera.point_cloud = True
        self.obs_config.front_camera.point_cloud = True
        self.obs_config.wrist_camera.point_cloud = True

        # 创建 RLBench 环境
        self.env = Environment(
            action_mode=self.action_mode,
            obs_config=self.obs_config,
            headless=headless,
            data_path='./rlbench_data'  # 数据保存路径
        )
        self.env.launch()

        # 获取任务
        self.task_name = task_name
        self.task = self.env.get_task(self._get_task_class(task_name))

        # 状态变量
        self.current_obs = None

    def _get_task_class(self, task_name):
        """将任务名称映射到RLBench任务类"""
        task_mapping = {
            'close_box': CloseBox,
            'open_box': OpenBox,
            'pick_and_lift': PickAndLift,
            'reach_target': ReachTarget,
            'stack_blocks': StackBlocks,
            'slide_block_to_target': SlideBlockToTarget,
            'push_buttons': PushButtons,
            # 添加更多任务映射
        }
        return task_mapping.get(task_name, CloseBox)

    def reset(self, randomize=False):
        # RLBench 的 reset 返回描述和观察
        descriptions, self.current_obs = self.task.reset()
        return self._process_observation(self.current_obs)

    def step(self, action):
        # 处理动作格式以适应 RLBench
        processed_action = self._process_action(action)
        self.current_obs, reward, done = self.task.step(processed_action)

        # 创建兼容的返回格式
        action_info = {
            'reward': reward,
            'task_success': terminate and reward > 0,
            'done': terminate
        }

        obs = self._process_observation(self.current_obs)
        return action_info

    def get_observation(self):
        if self.current_obs is None:
            self.reset()
        return self._process_observation(self.current_obs)

    def _process_observation(self, obs):
        # 将 RLBench 观察转换为 DROID 格式
        # processed_obs = {
        #     'robot_state': {
        #         'cartesian_position': self._extract_ee_pose(obs),
        #         'joint_positions': obs.joint_positions,
        #         'gripper_open': obs.gripper_open,
        #     },
        #     'timestamp': {},
        #     # 添加相机数据
        #     'wrist_rgb': obs.wrist_rgb,
        #     'front_rgb': obs.front_rgb,
        #     'overhead_rgb': obs.overhead_rgb,
        #     'left_shoulder_rgb': obs.left_shoulder_rgb,
        #     'right_shoulder_rgb': obs.right_shoulder_rgb,
        # }

        """处理观察数据为兼容格式"""
        processed_obs = {
            'robot_state': {
                'cartesian_position': self._extract_cartesian_position(obs),
                'joint_positions': obs.joint_positions,
                'gripper_position': 1.0 if obs.gripper_open else 0.0,
                'gripper_open': obs.gripper_open,
            },
            'image': {},
            'depth': {},
            'timestamp': {},
            'proprio': self._extract_proprioception(obs)
        }

        # 处理相机数据
        cameras = ['wrist', 'front', 'left_shoulder', 'right_shoulder', 'overhead']
        for camera in cameras:
            rgb_attr = f'{camera}_rgb'
            point_cloud_attr = f'{camera}_point_cloud'

            if hasattr(obs, rgb_attr):
                rgb_data = getattr(obs, rgb_attr)
                if rgb_data is not None:
                    processed_obs['image'][f'{camera}_rgb'] = rgb_data

            if hasattr(obs, point_cloud_attr):
                point_cloud = getattr(obs, point_cloud_attr)
                if point_cloud is not None:
                    # 使用点云的z坐标作为深度
                    depth_data = point_cloud[:, :, 2]
                    processed_obs['depth'][f'{camera}_depth'] = depth_data

        return processed_obs

    def _extract_cartesian_position(self, obs):
        """从gripper_pose提取笛卡尔位置和四元数"""
        if hasattr(obs, 'gripper_pose'):
            # gripper_pose是 [x, y, z, qx, qy, qz, qw]
            return obs.gripper_pose
        else:
            # 备用方法
            return np.concatenate([
                obs.gripper_pose[:3],  # 位置
                obs.gripper_pose[3:]  # 四元数
            ])

    def _extract_proprioception(self, obs):
        """提取本体感知数据"""
        cartesian_pos = self._extract_cartesian_position(obs)
        gripper_pos = 1.0 if obs.gripper_open else 0.0

        # 组合为 [x, y, z, qx, qy, qz, qw, gripper]
        proprio = np.concatenate([cartesian_pos, [gripper_pos]])
        return proprio

    def _process_action(self, action):
        # 将 DROID 动作转换为 RLBench 格式
        # action 是 [x, y, z, rot_x, rot_y, rot_z, gripper]
        if len(action) == 7:
            # 位置 + 欧拉角 + 夹爪
            pos = action[:3]
            euler = action[3:6]
            gripper = action[6]

            # 将欧拉角转换为四元数 (简化处理)
            # 在实际应用中应该使用正确的转换
            from scipy.spatial.transform import Rotation as R
            rot = R.from_euler('xyz', euler)
            quat = rot.as_quat()  # [x, y, z, w]

            # RLBench 期望的动作格式: [x, y, z, qx, qy, qz, qw, gripper]
            return np.concatenate([pos, quat, [gripper]])
        elif len(action) == 8:
            # 已经是正确格式
            return action
        else:
            raise ValueError(f"Unexpected action shape: {action.shape}")

    # def read_cameras(self):
    #     # 返回相机数据
    #     return {
    #         'wrist_rgb': self.current_obs.wrist_rgb if self.current_obs else None,
    #         'front_rgb': self.current_obs.front_rgb if self.current_obs else None,
    #     }
    def get_side_camera_extrinsics(self):
        """获取侧相机外参（简化版本）"""
        # 返回默认的外参矩阵
        return {
            'left_shoulder_camera': np.eye(4),
            'right_shoulder_camera': np.eye(4),
            'front_camera': np.eye(4)
        }

    def shutdown(self):
        self.env.shutdown()
