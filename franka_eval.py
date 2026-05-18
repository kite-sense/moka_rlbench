# import os
# import sys

# # =========================================================
# # 【强制环境配置】 放在所有 import 之前，防止 SegFault
# # =========================================================

# # 1. 强制使用 CPU 软件渲染 (解决 Signal 11 崩溃的核心)
# # os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

# # 2. 告诉 Mesa 驱动去哪里找库 (刚才建立软链接的目录)
# os.environ['LIBGL_DRIVERS_PATH'] = '/usr/lib/dri'

# # 3. 修正系统库路径，防止 Anaconda 的库冲突
# #    确保 /usr/lib/x86_64-linux-gnu 在最前面
# if 'LD_LIBRARY_PATH' not in os.environ:
#     os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/lib'
# else:
#     os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/lib:' + os.environ['LD_LIBRARY_PATH']

# print("Environment variables forced for Headless Rendering.")
# =========================================================

from rlbench.environment import Environment
import os
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
# from rlbench.environment import Environment
from rlbench.observation_config import CameraConfig, ObservationConfig

# import os  # NOQA
from absl import app, flags, logging  # NOQA
# from rlbench_env_wrapper import RLBenchEnvWrapper

# get the openai environment key
# import openai  # NOQA
if 'OPENAI_API_KEY' not in os.environ:
    raise RuntimeError(
        'OPENAI_API_KEY environment variable is not set. '
        'Export it before running: export OPENAI_API_KEY="sk-..."'
    )

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt  # NOQA
# plt.switch_backend('agg')  # To avoid the runtime error.

# from droid.robot_env import RobotEnv  # NOQA
from moka.drivers.new_driver import Driver  # NOQA
from moka.planners.visual_prompt_planner import VisualPromptPlanner  # NOQA
from moka.policies.franka_policy import FrankaMarkPolicy  # NOQA
from moka.utils.config_utils import load_config  # NOQA

# import pdb
# from rlbench.backend.spawn_boundary import BoundingBox
# from tqdm import tqdm
# from tqdm import trange
# from ip.utils.common_utils import *
# from ip.utils.data_proc import *
from ip_tasks import TASK_NAMES


FLAGS = flags.FLAGS

flags.DEFINE_string('config', './config/moka.yaml', 'Location of config file')
flags.DEFINE_string('data_dir', './data', 'Directory to save the data.')
# flags.DEFINE_string('task', 'open box', 'Name of the task.')
# flags.DEFINE_string('task', 'close box', 'Name of the task.')
# flags.DEFINE_string('task', 'pick the block and place it in the box', 'Name of the task.')
flags.DEFINE_string('task', 'push the yellow block to the white target, then stack the red block on top of the yellow block', 'Name of the task.')
# flags.DEFINE_string('task', 'close the box then open it', 'Name of the task.')
# flags.DEFINE_string('task', 'put the small laptop into the tray, then put the cup into the tray', 'Name of the RLBench task.')  # 改为RLBench任务名
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_boolean('debug', False, 'Debugging mode.')

flags.DEFINE_boolean('manual', False, 'Manually clicking the points.')
flags.DEFINE_boolean('skip_confirmation', False, 'Skip confirmation.')
flags.DEFINE_boolean('reuse_segmasks', False, 'Using computed segmasks.')
flags.DEFINE_boolean('incontext', False, 'Using in-context examples.')
flags.DEFINE_boolean('save_incontext', False, 'Saving in-context examples.')


# Table wiping
task_instructions = {
    # 'table_wiping':
    #     'Move the eyeglasses onto the yellow cloth and use the brush to sweep the blue snack package to the right side of the table.',  # NOQA
    #
    # 'ultrasound_cleaning':
    #     'Use the white ultrasound cleaner to clean the metal watch. The unstrasound cleaner has no lid and can be turned on by pressing the red button.',  # NOQA
    #
    # 'gift_preparation':
    #     'Make a pink gift box containing the perfurme bottle. Make sure to put some golden filler beneath the perfume.',  # NOQA
    #
    # 'laptop_packing':
    #     'Unplug the charge cable and close the lid of the laptop.',  # NOQA
    #
    # 'fur_removing':
    #     'Use the fur remover to swipe the white fur ball down.',  # NOQA
    #
    # 'grocery_bagging':
    #     'Put the white soap box and the chips bag into the brown paper bag in the correct order. Be careful that the soap is heavy and can smash the chips if it is on the top.',  # NOQA

    'close_box': 'Close the box lid.',
    'open_box': 'Open the box lid.',
    'pick_and_lift': 'Pick up the object and lift it.',
    'reach_target': 'Reach the target position.',

    'test':
        'test'

}

# TASK_NAMES = {
#     'lift_lid': TakeLidOffSaucepan,
#     'phone_on_base': PhoneOnBase,
#     'open_box': OpenBox,
#     'slide_block': SlideBlockToTarget,
#     'close_box': CloseBox,
#     'basketball': BasketballInHoop,
#     'buzz': BeatTheBuzz,
#     'close_microwave': CloseMicrowave,
#     'plate_out': TakePlateOffColoredDishRack,
#     'toilet_seat_down': ToiletSeatDown,
#     'toilet_seat_up': ToiletSeatUp,
#     'toilet_roll_off': TakeToiletRollOffStand,
#     'open_microwave': OpenMicrowave,
#     'lamp_on': LampOn,
#     'umbrella_out': TakeUmbrellaOutOfUmbrellaStand,
#     'push_button': PushButton,
#     'put_rubbish': PutRubbishInBin,
# }


def main(_):  # NOQA
    
    # import os
    # # 移除可能由 opencv 设置的 Qt 插件路径，防止干扰 PyRep
    # if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
    #     os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    # img_size = (1280, 720)
    img_size = (1280, 1000)
    obs_config.front_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size

    obs_config.front_camera.depth_in_meters = True
    obs_config.right_shoulder_camera.depth_in_meters = True
    obs_config.left_shoulder_camera.depth_in_meters = True
    obs_config.wrist_camera.depth_in_meters = True
    obs_config.overhead_camera.depth_in_meters = True

    obs_config.front_camera.intrinsics = True 
    obs_config.right_shoulder_camera.intrinsics = True

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(
            absolute_mode=True,
            collision_checking=False
        ),
        gripper_action_mode=Discrete()
    )
    env = Environment(action_mode,
                      './',
                      obs_config=obs_config,
                      headless=False)
    env.launch()

    config = load_config(FLAGS.config)

    # task = env.get_task(TASK_NAMES[FLAGS.task])
    # task = env.get_task(TASK_NAMES['large_container_gym'])
    task = env.get_task(TASK_NAMES['block_gym'])
    # task = env.get_task(TASK_NAMES['close_box_then_open'])
    # task = env.get_task(TASK_NAMES['pick_and_place_block'])

    if FLAGS.manual:
        planner = MouseClickPlanner(
            config=config,
            debug=FLAGS.debug,
        )
    else:
        planner = VisualPromptPlanner(
            config=config,
            debug=FLAGS.debug,
            skip_confirmation=FLAGS.skip_confirmation,
            reuse_segmasks=FLAGS.reuse_segmasks,
            use_incontext_examples=FLAGS.incontext,
            task_name=FLAGS.task,
        )

    # print("Constructing Moka-compatible camera info from RLBench...")
    # task.reset()
    # planner_cam_config = config.camera.planner
    # camera_name = f"{planner_cam_config.serial_number}_{planner_cam_config.view}"
    # try:
    #     sim_cam = task._task._scene._cam_front 
    # except AttributeError:
    #     from pyrep.objects.vision_sensor import VisionSensor
    #     sim_cam = VisionSensor('cam_front')
    # intrinsics = sim_cam.get_intrinsic_matrix() # 3x3 矩阵
    # extrinsics = sim_cam.get_matrix()

    # moka_camera_info = {
    #     'name': camera_name,
    #     'params': {
    #         'intrinsics': intrinsics,
    #         'extrinsics': extrinsics,
    #         # 关键补充：告诉 Moka 不需要再对深度图除以 1000，因为我们已经开了 depth_in_meters
    #         'depth_scale': 1.0 
    #     },
    #     'crop': planner_cam_config.crop  # 直接使用配置文件里的 crop 参数
    # }

    # planner.camera_info = moka_camera_info

    
    policy = FrankaMarkPolicy(
        config=config,
        debug=FLAGS.debug,
    )

    # 使用 RLBench 环境替代原来的 RobotEnv
    # env = RLBenchEnvWrapper(
    #     task_name=FLAGS.task,
    #     headless=True  # 设置为True用于训练，False用于调试
    # )

    # env = RobotEnv(
    #     action_space="cartesian_position",
    #     camera_kwargs=dict(
    #         varied_camera=dict(
    #             image=True,
    #             depth=True,
    #             pointcloud=False,
    #             concatenate_images=False,
    #         ),
    #     ),
    # )

    data_collector = Driver(
        env=task,
        policy=policy,
        planner=planner,
        task_instruction=FLAGS.task,
        # task_instruction=task_instructions.get(FLAGS.task, FLAGS.task),
        save_traj_dir=os.path.join(FLAGS.data_dir, 'saved'),
        save_data=True,
        config=config,
        debug=FLAGS.debug,
        skip_confirmation=FLAGS.skip_confirmation,
    )

    try:
        data_collector.collect_trajectory(reset_robot=True)
    finally:
        env.shutdown()


if __name__ == '__main__':
    app.run(main)
