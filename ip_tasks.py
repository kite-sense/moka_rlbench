"""
这里只包含了instant policy论文中使用的任务，后续扩展到整个RLBench所有的内置任务
TODO:
    1. 扩展到整个RLBench所有的内置任务，并支持自动添加
    2. 支持自动生成content中的描述文本(基本实现)
"""

# 参考 rl_bench_tasks.py 类似的结构，
# 但为了避免因某些任务类在当前 rlbench 版本中不存在而导致导入错误，
# 这里通过 getattr 动态获取类，找不到时赋值为 None。
from rlbench import tasks as rl_tasks
from pathlib import Path
import os


TASK_NAMES = {}

def add_all_tasks():
    """
    TODO: 自动添加所有内置任务到 TASK_NAMES 字典中
    """
    pass

def _safe_add(key, class_name):
    if not hasattr(rl_tasks, class_name):
        raise ValueError(f"Class {class_name} not found in rlbench.tasks.\n"
                         + "Check the class name!")
    TASK_NAMES[key] = getattr(rl_tasks, class_name)

# 根据原始文本列表构建映射（key 为 snake_case）
_safe_add('open_box', 'OpenBox')
_safe_add('close_jar', 'CloseJar')
_safe_add('toilet_seat_down', 'ToiletSeatDown')
_safe_add('close_microwave', 'CloseMicrowave')
_safe_add('phone_on_base', 'PhoneOnBase')
_safe_add('lift_lid', 'TakeLidOffSaucepan')
_safe_add('umbrella_out', 'TakeUmbrellaOutOfUmbrellaStand')
_safe_add('slide_block', 'SlideBlockToTarget')
_safe_add('push_button', 'PushButton')
_safe_add('basketball', 'BasketballInHoop')
_safe_add('meat_on_grill', 'MeatOnGrill')
_safe_add('flip_switch', 'PressSwitch')
_safe_add('buzz', 'BeatTheBuzz')
_safe_add('plate_out', 'TakePlateOffColoredDishRack')
_safe_add('close_laptop', 'CloseLaptopLid')
_safe_add('close_box', 'CloseBox')
_safe_add('open_jar', 'OpenJar')
_safe_add('toilet_seat_up', 'ToiletSeatUp')
_safe_add('meat_off_grill', 'MeatOffGrill')
_safe_add('open_microwave', 'OpenMicrowave')
_safe_add('toilet_roll_off', 'TakeToiletRollOffStand')
_safe_add('put_rubbish', 'PutRubbishInBin')
_safe_add('put_umbrella', 'PutUmbrellaInUmbrellaStand')
_safe_add('lamp_on', 'LampOn')
# TODO: 添加其他任务
_safe_add('large_container_gym', 'LargeContainerGym')
_safe_add('block_gym', 'BlockGym')
_safe_add('close_box_then_open', 'CloseBoxThenOpen')
_safe_add('pick_and_place_block', 'PickAndPlaceBlock')

# 可选：方便展示的可读任务名
IP_TASK_LABELS = {k: k.replace('_', ' ').title() for k in TASK_NAMES.keys()}
__all__ = ['TASK_NAMES', 'IP_TASK_LABELS']

####################################################################################################################

def _get_default_task_name(key: str) -> str:
    cls = TASK_NAMES.get(key)
    if cls is not None:
        try:
            class_name = cls.__name__
            import re
            words = re.findall(r'[A-Z][a-z]*', class_name)
            if words:
                # 首字母大写，其余单词小写
                label = words[0].capitalize() + ' ' + ' '.join(w.lower() for w in words[1:])
            else:
                label = IP_TASK_LABELS.get(key, key.replace('_', ' ').title())
        except Exception:
            label = IP_TASK_LABELS.get(key, key.replace('_', ' ').title())
    else:
        label = IP_TASK_LABELS.get(key, key.replace('_', ' ').title())
    return f"{label}."

def dump_sample_as_json(path: str):
    """
    将当前 TASK_NAMES 的条目导出为 data/RLbench_samples.json
    [
    {
        "id": int,
        "task_name": str,
        "content": str
        },
        ...
    ]
    content 生成规则：
        1. 如果能成功实例化任务类并且实例有 init_episode 方法，则调用 init_episode(0) 并取返回列表的第一个字符串作为 content。
        2. 否则使用可读的任务描述（IP_TASK_LABELS 生成的短语）。
    """
    # from rlbench.environment import Environment
    # from rlbench.action_modes.action_mode import MoveArmThenGripper
    # from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
    # from rlbench.action_modes.gripper_action_modes import Discrete
    import traceback
    import json
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # action_mode = MoveArmThenGripper(arm_action_mode=EndEffectorPoseViaIK(), gripper_action_mode=Discrete())
    # env = Environment(action_mode=action_mode, obs_config=None, headless=True)
    # env.launch()

    from pyrep import PyRep
    from pyrep.robots.arms.panda import Panda
    from pyrep.robots.end_effectors.panda_gripper import PandaGripper
    from rlbench.backend.robot import Robot
    from rlbench.tasks.close_box import CloseBox
    import os

    pr = PyRep()
    # 指定 RLBench 的主场景文件（相对仓库根目录）
    task_design_path = os.environ.get(
        'RLBENCH_TASK_DESIGN_TTT',
        os.path.join(os.path.dirname(__file__), '..', 'rlbench', 'task_design.ttt')
    )
    pr.launch(task_design_path, headless=True, responsive_ui=True)
    pr.start()
    # 创建 Robot 包装器
    robot = Robot(Panda(), PandaGripper())

    samples = []
    for i, key in enumerate(sorted(TASK_NAMES.keys()), start=1):
        # default：使用类名作为任务描述
        content = _get_default_task_name(key)
        # 如果存在 init_episode 方法，则尝试调用以获取更具体的描述
        cls = TASK_NAMES.get(key)
        if cls is not None:
            try:
                # 尝试实例化任务类并调用 init_episode(0)
                inst = cls(pr, robot)
                if hasattr(inst, 'init_episode') and callable(getattr(inst, 'init_episode')):
                    try:
                        descs = inst.init_episode(0)
                        content = descs[0] # 取第一个描述
                    except Exception:
                        print(f"Warning: Could not call init_episode(0) for task '{key}'.")
                        traceback.print_exc()
            except Exception:
                # 无法实例化则忽略，使用 label 作为 content
                print(f"Warning: Could not instantiate {cls} for task '{key}'. Using default label as content.")
                traceback.print_exc()

        # try:
        #     task = env.get_task(cls)   # cls 来自 TASK_NAMES[key]
        #     try:
        #         descs = task.init_episode(0)
        #         print('init_episode ->', descs)
        #     except Exception:
        #         traceback.print_exc()
        # except Exception as e:
        #     print(f"Warning: Could not get task instance for '{key}': {e}")

        samples.append({"id": i, "task_name": key, "content": content})

    # env.shutdown()
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)
    return out_path

def main():
    import argparse
    default_path = Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'test_RLbench_samples.json'
    parser = argparse.ArgumentParser(description='Dump TASK_NAMES -> json sample file')
    parser.add_argument('--out_dir', type=str, default=str(default_path),
                        help='output json file path (default: data/RLbench_samples.json)')
    args = parser.parse_args()
    out_file = dump_sample_as_json(args.out_dir)
    print(f"Saved {len(TASK_NAMES)} samples to {out_file}")

if __name__ == '__main__':
    main()

