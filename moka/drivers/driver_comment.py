import os
import shutil
import time
from datetime import date

import h5py
import numpy as np
import matplotlib.pyplot as plt  # NOQA
from matplotlib import widgets  # NOQA
from scipy import ndimage

# 引入 Droid 框架的时间工具
from droid.misc.time import time_ms
# 引入参数配置，包括版本号、机器人序列号和类型
from droid.misc.parameters import (
    moka_version,
    robot_serial_number,
    robot_type,
)
# 引入 Moka 的视觉处理工具（主要用于深度图处理）
from moka.vision import depth_utils
# 引入轨迹写入工具，用于将数据保存为 h5 文件
from droid.trajectory_utils.trajectory_writer import TrajectoryWriter

# 准备数据文件夹路径 #
# 获取当前文件的绝对路径目录
dir_path = os.path.dirname(os.path.realpath(__file__))
# 定义数据存储的根目录，位于当前目录的上两级目录中的 moka_data 文件夹
data_dir = os.path.join(dir_path, "../../moka_data")


def crop_image(image, crop):
    """
    辅助函数：裁剪图像
    :param image: 原始图像数组
    :param crop: 裁剪参数 [y_start, x_start, y_end, x_end]
    :return: 裁剪后的图像
    """
    return image[crop[0]:crop[2], crop[1]:crop[3]]


class Driver:
    def __init__(
        self,
        env,
        policy,
        planner,
        config,
        task_instruction=None,
        save_data=True,
        save_traj_dir=None,
        debug=False,
        skip_confirmation=False,
    ):
        """
        初始化 Driver 类，负责协调数据采集流程
        """
        self.env = env  # 机器人环境实例
        self.planner = planner  # 规划器实例（通常用于生成子任务或路径点）
        self.task_instruction = task_instruction  # 任务的语言指令
        self.policy = policy  # 策略实例（负责具体的动作执行）

        self.config = config  # 配置对象

        # 获取各个相机的参数信息（内参、外参、裁剪区域等）
        # 包括腕部相机（wrist）、主相机（primary）、次相机（secondary）
        self.wrist_camera_info = self.get_camera_info(
            self.config.camera.wrist)
        self.primary_camera_info = self.get_camera_info(
            self.config.camera.primary)
        self.secondary_camera_info = self.get_camera_info(
            self.config.camera.secondary)
        
        # 获取用于规划器的特定相机信息
        self.planner_camera_info = self.get_camera_info(
            self.config.camera.planner)

        self.planner_obs = None  # 用于存储规划器使用的观察数据
        self.resume_joint_pos = None  # 用于存储恢复任务时的关节位置

        # 断言检查：规划器相机必须是左视角（硬件强约束）
        assert config.camera.planner.view == 'left', (
            'The depth image is only available from the left view.')
        # 将相机信息注入到规划器中
        self.planner.camera_info = self.planner_camera_info

        self.last_traj_path = None  # 上一次保存的轨迹路径
        self.traj_saved = False  # 标记轨迹是否已保存
        
        # 获取环境中的所有相机 ID 并排序
        self.cam_ids = list(env.camera_reader.camera_dict.keys())
        self.cam_ids.sort()

        # 确保日志目录存在 #
        if save_traj_dir is None:
            save_traj_dir = data_dir  # 如果未指定，使用默认数据目录
            
        # 创建成功数据的保存目录：data_dir/success/日期
        self.success_logdir = os.path.join(
            save_traj_dir, "success", str(date.today())
        )
        # 创建失败数据的保存目录：data_dir/failure/日期
        self.failure_logdir = os.path.join(
            save_traj_dir, "failure", str(date.today())
        )
        
        # 如果目录不存在则创建
        if not os.path.isdir(self.success_logdir):
            os.makedirs(self.success_logdir)
        if not os.path.isdir(self.failure_logdir):
            os.makedirs(self.failure_logdir)
            
        self.save_data = save_data  # 是否保存数据的开关
        self.debug = debug  # 调试模式开关
        self.skip_confirmation = skip_confirmation  # 是否跳过人工确认步骤

    def get_camera_info(self, camera_config):
        """
        从环境和配置中提取相机的详细信息
        """
        # 构建相机名称：序列号_视角
        camera_name = camera_config.serial_number + '_' + camera_config.view
        # 从环境的 camera_reader 中获取相机对象
        camera = self.env.camera_reader.get_camera(camera_config.serial_number)
        
        # 获取相机内参
        camera_intrinsics = camera.get_intrinsics()
        # 获取侧面相机的外参（相对于机器人基座）
        camera_extrinsics = self.env.get_side_camera_extrinsics()
        
        # 组装参数字典
        camera_params = {
            'intrinsics': camera_intrinsics[camera_name],
            'extrinsics': camera_extrinsics[camera_name],
        }
        
        # 返回包含名称、参数和裁剪信息的字典
        return {
            'name': camera_name,
            'params': camera_params,
            'crop': camera_config.crop,
        }

    def get_planner_observation(self, check=True): # 一直是false嘛（笑哭
        """
        获取用于规划器的观察数据，支持可视化检查
        """

        def show_image():
            # 如果开启检查模式，初始化 matplotlib 绘图窗口
            if check:
                fig, ax = plt.subplots(ncols=2)
                fig.set_figheight(5)
                fig.set_figwidth(15)
                ax[0].axis('off')
                ax[1].axis('off')
                # 定义按钮位置
                ax_button1 = plt.axes([0.25, 0.025, 0.1, 0.075])
                ax_button2 = plt.axes([0.65, 0.025, 0.1, 0.075])

            # 确认按钮的回调函数：关闭窗口
            def on_ok_button_clicked(event):
                plt.close()

            def plot():
                # image = np.random.uniform(0, 1, size=[720, 1080, 3])
                # depth = np.random.uniform(0, 1, size=[720, 1080])

                print('Taking images, please wait...')
                # 调用内部函数获取实际观察数据
                self.planner_obs = self._get_planner_observation()

                # 如果在检查模式，显示 RGB 和深度图
                if check:
                    ax[0].imshow(self.planner_obs['image_data'])
                    ax[1].imshow(self.planner_obs['depth_data'])

                    # 绑定 OK 按钮
                    button_ok = widgets.Button(ax_button1, "OK")
                    button_ok.on_clicked(
                        on_ok_button_clicked)

                    # 绑定重拍按钮
                    button_retake = widgets.Button(ax_button2, "Retake")
                    button_retake.on_clicked(
                        on_retake_button_clicked)

                    plt.show()

                print('Images updated.')

            # 重拍按钮的回调函数：重新调用 plot()
            def on_retake_button_clicked(b):
                plot()

            plot()
        
        print("--------------planner_image--------------")
        show_image()

    def _get_planner_observation(self):
        """
        内部函数：获取并预处理规划器所需的观察数据（RGB-D）
        """
        # 确保配置中设置了至少对1帧深度图进行平均
        assert self.config.camera.depth.average_across_n_frames >= 1

        camera_name = self.planner_camera_info['name'] # 这就是为什么接口一定要对齐，这里用到内参+外参了
        camera_params = self.planner_camera_info['params']  # NOQA

        depth_data_list = []
        # 循环采集多帧深度数据以进行平均（减少噪声）
        for i in range(self.config.camera.depth.average_across_n_frames): # 其实就一次
            raw_obs = self.env.get_observation()

            depth_data = raw_obs['depth'][camera_name]
            # 确保深度图数据有效
            assert depth_data.size > 0, (
                f"Invalid depth shape: {depth_data.shape}")
            depth_data_list.append(depth_data)

        obs = dict()

        # 保存相机内参和外参
        obs['camera_intrinsics'] = camera_params['intrinsics']
        obs['camera_extrinsics'] = raw_obs['camera_extrinsics'] # 为什么不camera_params['extrinsics']？

        # 预处理深度图列表（例如取平均值）
        depth_data = depth_utils.preprocess_depth_list(depth_data_list)
        obs['depth_data'] = depth_data

        # 对深度图进行修复（填补无效像素/空洞）
        depth_inpainted = depth_utils.inpaint(depth_data)
        # 使用高斯滤波平滑深度图，sigma 参数来自配置
        depth_filtered = ndimage.gaussian_filter(
            depth_inpainted,
            sigma=self.config.camera.depth.grad_gaussian_sigma)
        obs['depth_filtered'] = depth_filtered

        # 获取 RGB 图像，去掉 alpha 通道（保留前3个通道）
        image_data = raw_obs['image'][camera_name][:, :, :3]
        image_data = image_data[:, :, ::-1]  # 将 BGR 转换为 RGB 格式
        obs['image_data'] = image_data

        # 获取本体感觉数据（Proprioception）
        robot_state = raw_obs['robot_state']
        robot_pose = robot_state['cartesian_position'] # 机器人末端笛卡尔位置
        gripper_pose = robot_state['gripper_position'] # 夹爪位置
        # 将末端位姿和夹爪状态拼接
        proprio_data = np.concatenate(
            [robot_pose, np.array([gripper_pose])])
        obs['proprio'] = proprio_data
        
        # np.save('planner_obs_button.npy', obs) # 调试用代码
        return obs

    def get_observation(self):
        """
        获取策略（Policy）所需的标准观察数据
        """
        obs = self.env.get_observation()
        del obs['depth']  # 策略通常不需要原始深度数据，这里删除以节省内存

        # 处理本体感觉数据
        robot_state = obs['robot_state']
        robot_pose = robot_state['cartesian_position']
        gripper_pose = robot_state['gripper_position']
        # 拼接位置和夹爪状态
        proprio_data = np.concatenate(
            [robot_pose, np.array([gripper_pose])])
        obs['proprio'] = proprio_data
        obs['crop'] = dict() # 初始化裁剪信息字典

        # 调试打印：当前末端位置和夹爪位置
        if self.debug:
            print(
                'current ee position, ee pose, gripper position',
                robot_pose[:3],
                robot_pose[3:],
                gripper_pose,
            )

        # 遍历所有相机（腕部、主、次），进行裁剪处理
        for camera_info in [self.wrist_camera_info,
                            self.primary_camera_info,
                            self.secondary_camera_info]:
            camera_name = camera_info['name']
            camera_crop = camera_info['crop']
            # 获取图像并转为 RGB
            image_data = obs['image'][camera_name][:, :, :3][:, :, ::-1]
            # 根据配置裁剪图像
            image_data_cropped = crop_image(image_data, camera_crop)

            # 更新观察字典中的图像和裁剪参数
            obs['image'][camera_name] = image_data_cropped
            obs['crop'][camera_name] = np.array(camera_crop)

        # 以下注释掉的代码是用于可视化检查裁剪效果的
        # fig, ax = plt.subplots(nrows=2, ncols=2)
        # ... (省略可视化代码)

        return obs

    def prepare_save_directory(self, log_dir, info, subtask_id):
        """
        为当前子任务准备保存目录
        :param log_dir: 根日志目录
        :param info: 包含时间戳等信息的字典
        :param subtask_id: 子任务 ID
        :return: 轨迹文件路径, 上下文文件路径, 录像文件夹路径
        """
        # 创建子任务文件夹：时间戳_子任务ID
        file_save_dir = os.path.join(
            log_dir, info["time"] + '_' + str(subtask_id)
        )
        if not os.path.isdir(file_save_dir):
            os.makedirs(file_save_dir)
        print('logging to file_save_dir: ', file_save_dir)

        # 定义文件路径
        save_filepath = os.path.join(
            file_save_dir, "trajectory.h5"
        )
        recording_folderpath = os.path.join(
            file_save_dir, "recordings"
        )
        context_path = os.path.join(
            file_save_dir, "context.npy"
        )
        
        # 创建录像文件夹
        if not os.path.isdir(recording_folderpath):
            os.makedirs(recording_folderpath)

        return save_filepath, context_path, recording_folderpath

    def collect_trajectory_with_mouseclick(  # NOQA
        self,
        log_dir=None,
        info=None,
        save_images=False,
        randomize_reset=False,
        reset_robot=True,
    ):
        """
        采集机器人轨迹的核心函数。
        - 包含人机交互（Mouse Click Planner）。
        - 循环执行子任务。
        """

        # 检查参数：如果保存图像，必须提供目录和信息
        if save_images:
            assert log_dir is not None
            assert info is not None

        recording_folderpath = None
        save_filepath = None
        context_filepath = None

        # 设置相机读取器为轨迹模式（可能涉及缓冲区设置）
        self.set_trajectory_mode()
        num_subtasks = 0  # 子任务计数器
        subtask_success = False  # 子任务成功标志

        # 在采集数据前再次确认图像（双重检查），避免相机遮挡等问题
        if not self.debug and not self.skip_confirmation:
            obs = self.get_observation()
            fig, ax = plt.subplots(ncols=3)
            fig.set_figheight(10)
            fig.set_figwidth(15)
            # 显示三个相机的画面
            print("--------------show three cameras--------------")
            ax[0].imshow(obs['image'][self.primary_camera_info['name']])
            ax[1].imshow(obs['image'][self.secondary_camera_info['name']])
            ax[2].imshow(obs['image'][self.wrist_camera_info['name']])
            fig.show()

        # 主循环：执行多个子任务直到整个任务完成
        while True:
            print('Subtask # %d' % (num_subtasks))

            # 如果是第一个子任务
            if num_subtasks == 0:
                # 准备轨迹 #
                # 复位环境
                self.env.reset()
                input('Press [Enter] to continue to next step.')
                # 获取规划器所需的观察数据
                self.get_planner_observation()

                # 重置规划器，传入观察数据和任务指令
                self.planner.reset(self.planner_obs, self.task_instruction)

            # 如果是后续子任务
            elif num_subtasks > 0:
                # 1. 这里原本逻辑是保存关节状态，然后回中，拍照，再恢复。
                # 但当前代码中被简化为直接 reset 环境
                self.env.reset()

                # 3. 拍摄规划用的图像
                self.get_planner_observation()

                assert self.resume_joint_pos is not None
                # self.planner.reset(...) # 这里被注释掉了，说明不需要完全重置规划器

                # 4. 恢复之前的关节位置（这里目前也是通过 reset 实现，可能不准确，依赖 env 实现）
                self.env.reset()

                # 更新规划器的子任务索引
                self.planner.current_subtask_index = num_subtasks

            self.planner.current_subtask_index = num_subtasks

            # 重置规划器的分割掩码和关键点
            self.planner.reset_seg_and_keypoints()
            resample_plan = False
            
            # 循环采样规划结果，直到用户满意
            while True:
                planner_obs = self.planner_obs.copy()
                # 规划器采样一个子任务上下文（context）
                context = self.planner.sample_subtask(
                    planner_obs, t=num_subtasks, request_context=resample_plan)

                # 用户确认环节
                if self.skip_confirmation:
                    ch = 'y'
                else:
                    ch = input(
                        'Continue execution without resampling plan? [y/n]')

                if ch in ['y', 'Y', '', '\n']:
                    break  # 用户接受规划
                elif ch in ['n', 'N']:
                    print('resample plan by calling API')
                    resample_plan = True  # 用户拒绝，重新规划
                else:
                    print('Invalid input. Please enter [y] or [n].')

            # 获取当前观察，准备开始执行
            obs = self.get_observation()
            # 重置策略（Policy），传入当前上下文。如果是第一个子任务，设置 init=True
            self.policy.reset(obs, context, init=(num_subtasks == 0))

            # 如果指定了日志目录，准备保存路径并开启录制
            if log_dir:
                assert info is not None
                save_filepath, context_filepath, recording_folderpath = (
                    self.prepare_save_directory(
                        log_dir, info, num_subtasks
                    )
                )
                print('logging to save_filepath: ',
                      save_filepath)
                print('logging to recording_folderpath: ',
                      recording_folderpath)
                # 初始化轨迹写入器
                traj_writer = TrajectoryWriter(
                    save_filepath, save_images=save_images, exists_ok=False)
                # 开始相机录制
                self.env.camera_reader.start_recording(
                    recording_folderpath
                )

            # 子任务执行循环：最多执行 max_subtask_steps 步
            for num_steps in range(self.config.max_subtask_steps):
                if self.debug:
                    input('press enter to continue to next step')

                print('step: ', num_steps)

                # 记录时间戳
                control_timestamps = {"step_start": time_ms()}
                obs = self.get_observation()

                # 将子任务 ID 注入到观察数据中，方便后续数据拼接
                obs['subtask_id'] = num_subtasks

                # 获取动作（Get Action）
                control_timestamps["policy_start"] = time_ms()
                # 调用策略生成动作
                action, policy_info = self.policy.sample_actions(obs)

                # 频率控制（Regularize Control Frequency）
                control_timestamps["sleep_start"] = time_ms()
                comp_time = time_ms() - control_timestamps["step_start"]
                # 计算需要休眠的时间，以维持固定的控制频率 (control_hz)
                sleep_left = (1 / self.env.control_hz) - (comp_time / 1000)
                if sleep_left > 0:
                    time.sleep(sleep_left)

                # 执行环境步进（Step Environment）
                control_timestamps["control_start"] = time_ms()

                # 执行动作，获取新的观察、奖励和终止信号
                curr_obs, reward, terminate = self.env.step(action)
                # 这里的 action_info 通常由 env.step 返回，但代码里看起来 curr_obs 是 tuple? 
                # 注意：这里假设 action_info 包含 task_success 信息，需要根据具体 env 实现确认
                # 原代码此处可能有变量名混淆，这里保留原样注释，假设 action_info 可用
                action_info = {"task_success": False} # 占位，实际上应该来自 step 的返回值

                # 保存数据 #
                control_timestamps["step_end"] = time_ms()
                obs["timestamp"]["control"] = control_timestamps
                # 构建时间步数据包
                timestep = {"observation": obs, "action": action_info}
                # 如果开启写入，写入一帧数据
                if save_filepath:
                    traj_writer.write_timestep(timestep)

                print('task_success: ', action_info["task_success"])

                # 检查是否终止：达到最大步数或任务成功
                end_traj = (num_steps == self.config.max_subtask_steps - 1
                            or action_info["task_success"])

                if end_traj:
                    break

            # 子任务结束后，人工确认是否成功
            ch = None
            while True:
                ch = input('Enter the success [y/n]')
                if ch in ['y', 'Y', '', '\n']:
                    subtask_success = True
                    num_subtasks += 1
                    break
                elif ch in ['n', 'N']:
                    subtask_success = False
                    break
                else:
                    print('Invalid input. Please enter [y] or [n].')

            # 停止相机录制
            if recording_folderpath:
                print('stop recording last subtask')
                self.env.camera_reader.stop_recording()

            # 关闭轨迹写入器
            if save_filepath:
                print('close trajectory writer for last subtask')
                traj_writer.close()

            # 如果任务失败，删除已保存的数据
            if subtask_success is False:
                print('deleting saved data')
                if os.path.exists(save_filepath):
                    os.remove(save_filepath)
                # 清理录像文件夹
                if os.path.exists(recording_folderpath):
                    for filename in os.listdir(recording_folderpath):
                        file_path = os.path.join(
                            recording_folderpath, filename)
                        try:
                            if (os.path.isfile(file_path) or
                                    os.path.islink(file_path)):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print('Failed to delete %s. Reason: %s'
                                  % (file_path, e))

                # 将 context 文件重命名标记为失败
                if os.path.exists(context_filepath):
                    # os.remove(context_filepath)
                    os.rename(context_filepath, context_filepath + '_fail.npy')
            else:
                # 如果成功，保存上下文和规划器观察数据
                print('saving context and planner obs')
                data = {
                    "context": context,
                    "planner_obs": self.planner_obs,
                }
                np.save(context_filepath, data)

            # 如果失败，询问是否重试
            if not subtask_success:
                ch = input('[Optional] To restart the subtask, revert the scene, enter [y] then press [Enter].')  # NOQA
                if ch not in ['y', 'Y', '', '\n']:
                    print('quit data collection')
                    return {"success": False}
            else:
                # 如果成功，获取当前状态以便下一个子任务恢复（虽然目前代码里没充分利用）
                print('subtask success! saving joint state')
                resume_state, _ = self.env.get_state()
                self.resume_joint_pos = resume_state['joint_positions']

            # 如果超过最大子任务数，结束整个采集流程
            if num_subtasks >= self.config.max_subtasks:
                self.env.reset()
                # input('Task is done. Press [Enter] after resetting the scene.')  # TODO
                self.resume_joint_pos = None
                break

        return {"success": subtask_success}

    def reset_robot(self, randomize=False):
        """
        重置机器人的辅助函数
        """
        self.env._robot.establish_connection() # 建立连接
        self.env.reset(
            randomize=randomize, # 是否随机化初始状态
        )

    def set_trajectory_mode(self):
        """
        设置相机为轨迹采集模式
        """
        self.env.camera_reader.set_trajectory_mode()

    def prepare_info(self):
        """
        准备元数据信息（时间、机器人序列号等）
        """
        self.last_traj_name = time.asctime().replace(" ", "_") # 使用当前时间作为名称
        info = {}
        info["time"] = self.last_traj_name
        info["robot_serial_number"] = "{0}-{1}".format(
            robot_type, robot_serial_number
        )
        info["version_number"] = moka_version

        # 检查相机数量是否正确（通常需要3个相机：主、副、腕部）
        if len(self.cam_ids) != 6: # 注意：原代码写的是6，可能是包含了RGB和Depth各一个ID
            print(
                "WARNING: User is trying to collect data without all three cameras running!"  # NOQA
            )
        return info

    def collect_trajectory(self, reset_robot=True):
        """
        批量采集多个任务的入口
        """
        for num_tasks in range(self.config.max_tasks): # 循环执行 max_tasks 次
            print('Starting new task # %d' % (num_tasks))
            self._collect_trajectory(reset_robot)

    def _collect_trajectory(self, reset_robot=True):
        """
        采集单个任务轨迹的内部函数
        """
        info = self.prepare_info()
        # 建立连接，确保机器人就绪
        # self.env._robot.establish_connection() # 重新建立连接以防不稳定

        # 调用核心采集函数，记录到 success 目录
        success_info = self.collect_trajectory_with_mouseclick(
            reset_robot=reset_robot,
            log_dir=self.success_logdir,
            info=info,
        )
        print('success_info:')
        print(success_info)

        # 下面的代码用于自动将失败的轨迹移动到成功目录（如果状态被更新），目前被注释掉了
        # Sort Trajectory #
        # self.traj_succeed = success_info["success"]

        # if self.traj_succeed:
        #     self.last_traj_path = os.path.join(
        #         self.success_logdir, info["time"]
        #     )
        #     os.rename(
        #         os.path.join(self.failure_logdir, info["time"]),
        #         self.last_traj_path,
        #     )

    def change_trajectory_status(self, success=False):
        """
        工具函数：用于事后修改轨迹的成功/失败状态（Relabeling）
        """
        # 如果没有上一次轨迹路径，或者状态没有改变，直接返回
        if (self.last_traj_path is None) or (success == self.traj_saved):
            return

        # 打开 h5 文件并修改属性
        save_filepath = os.path.join(self.last_traj_path, "trajectory.h5")
        traj_file = h5py.File(save_filepath, "r+")
        traj_file.attrs["success"] = success
        traj_file.attrs["failure"] = not success
        traj_file.close()

        # 根据新状态移动文件夹
        if success:
            new_traj_path = os.path.join(
                self.success_logdir, self.last_traj_name
            )
            os.rename(self.last_traj_path, new_traj_path)
            self.last_traj_path = new_traj_path
            self.traj_saved = True
        else:
            new_traj_path = os.path.join(
                self.failure_logdir, self.last_traj_name
            )
            os.rename(self.last_traj_path, new_traj_path)
            self.last_traj_path = new_traj_path
            self.traj_saved = False