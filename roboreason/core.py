
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def load_video_frames(video_path):
    import cv2
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frames.append(frame)
    cap.release()
    return frames


CURRENT_MODEL = None


def generate(
    model: str,
    task_description: str,
    video_paths: list = None,
    video_path: str = None,
    ###### optionl, in case you want to give two videos with different views instead of one video that has a single view or has combined side-by-side views
    video_view_external_paths: list = None, 
    video_view_wrist_paths: list = None,
    video_view_external_path: str = None, 
    video_view_wrist_path: str = None,
    ######
    ###### optionl, in case you want to give frames instead of video paths
    video_frames: list = None,
    ######
    view_type_per_video: list = None,
    view_type: str = None,
    num_reasoning_frames: int = 10,
    context_window: str = None,
    ###### for API-based models
    key: str = None,
    ######
    ###### optional, for local models
    model_path: str = None,
    ######
):
    # 
    global CURRENT_MODEL
    # 
    if CURRENT_MODEL is not None and CURRENT_MODEL != model:
        print(f"Switching model from {CURRENT_MODEL} → {model}.")
        if not 'gpt' in model and not 'gemini' in model:
            print(f"Unloading previous model {CURRENT_MODEL} from GPU to free up memory...")
            # 
            if CURRENT_MODEL == "robometer":
                from roboreason.robometer.roboreason_robometer import unload_model as unload_robometer
                unload_robometer()
            # 
            elif CURRENT_MODEL == "sole-r1":
                from roboreason.sole import unload_model as unload_sole
                unload_sole()
            # 
            elif CURRENT_MODEL == "roboreward":
                from roboreason.roboreward import unload_model as unload_roboreward
                unload_roboreward()
            # 
            elif CURRENT_MODEL == "topreward":
                from roboreason.topreward import unload_model as unload_topreward
                unload_topreward()
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    # 
    CURRENT_MODEL = model
    # 
    if 'gpt' in model or 'gemini' in model:
        assert key is not None, 'API key must be provided for OpenAI and Google models'
    # 
    single_video = False
    if video_frames is None:
        if video_paths is None and video_path is not None:
            video_paths = [video_path]
            single_video = True
            # 
            if view_type_per_video is None and view_type is not None:
                view_type_per_video = [view_type] * len(video_paths)
        # 
        else:
            if video_view_external_paths is None and video_view_external_path is not None:
                video_view_external_paths = [video_view_external_path]
                single_video = True
            if video_view_wrist_paths is None and video_view_wrist_path is not None:
                video_view_wrist_paths = [video_view_wrist_path]
                single_video = True
            # 
            if view_type_per_video is None and view_type is not None:
                view_type_per_video = [view_type] * max([len(video_view_external_paths or []), len(video_view_wrist_paths or [])])
        # 
    else:
        # if video_frames[0] is a list of frames, then we have multiple videos worth of frames. if video_frames is a single list of frames, then we have one video worth of frames.
        if isinstance(video_frames[0], list):
            single_video = False
        else:
            single_video = True
    ############ EXTRACT VIDEO FRAMES FOR ALL VIDEOS AS A LIST OF LISTS    
    # 
    if video_frames is None:
        if video_paths is None and video_view_external_paths is not None and video_view_wrist_paths is not None:
            # concatenate external and wrist view videos side by side and use that as input to the model 
            videos = []
            for video_idx in range(len(video_view_external_paths)):
                frames_external = load_video_frames(video_view_external_paths[video_idx])
                frames_wrist = load_video_frames(video_view_wrist_paths[video_idx])
                if not len(frames_external) == len(frames_wrist):
                    raise ValueError(f"Number of frames in external view video {len(frames_external)} does not match number of frames in wrist view video {len(frames_wrist)} for video_idx {video_idx}")
                frames_combined = [np.concatenate((frames_external[i], frames_wrist[i]), axis=1) for i in range(len(frames_external))]
                videos.append(frames_combined)
            view_type_per_video = ['external and wrist'] * len(videos)
        else:
            if video_paths is None and video_view_external_paths is not None and video_view_wrist_paths is None:
                video_paths = video_view_external_paths
                view_type_per_video = ['external'] * len(video_view_external_paths)
                print(f"Using videos from video_view_external_paths with view type 'external'")
            elif video_paths is None and video_view_external_paths is None and video_view_wrist_paths is not None:
                video_paths = video_view_wrist_paths
                view_type_per_video = ['wrist'] * len(video_view_wrist_paths)
                print(f"Using videos from video_view_wrist_paths with view type 'wrist'")
            elif video_paths is None:
                raise ValueError("video_paths cannot be None if video_view_external_paths and video_view_wrist_paths are both not provided")
            # 
            videos = []
            for video_path in video_paths:
                frames = load_video_frames(video_path)
                videos.append(frames)
    else:
        if single_video:
            videos = [video_frames]
        else:
            videos = video_frames
    # 
    ############ DOWNSAMPLE TO NUM_REASONING_FRAMES (e.g. 10) REASONING FRAMES PER VIDEO
    downsampled_videos = []
    downsample_idx_list_list = []
    for video_idx in range(len(videos)):
        downsample_idx_list = np.linspace(0, len(videos[video_idx])-1, num=num_reasoning_frames, dtype=int)
        downsampled_videos.append([videos[video_idx][i] for i in downsample_idx_list])
        downsample_idx_list_list.append(downsample_idx_list.tolist())
    # 
    # 
    if not model in ['sole-r1']:
        frame_height, frame_width = downsampled_videos[0][0].shape[:2]
        if 'test_videos' in video_path:
            if frame_width == 2*frame_height:
                for video_idx in range(len(downsampled_videos)):
                    frames_final=[]
                    for i in range(len(downsampled_videos[video_idx])):
                        frames_final.append(downsampled_videos[video_idx][i][:, :downsampled_videos[video_idx][i].shape[1]//2, :])
                    # 
                    downsampled_videos[video_idx] = frames_final
    ############ GENERATE REWARD AND REASONING TRACES FOR EACH VIDEO
    # input_data = {
    #     'task_description': task_description,
    #     'video_path': video_path,
    #     'context_window': context_window,
    #     'num_reasoning_frames': num_reasoning_frames,
    #     'view_type_per_video': view_type_per_video
    # }
    success_probs = None
    reasoning_traces = None
    rewards = None
    full_rewards = None
    full_reasoning_traces = None
    if 'gpt' in model or 'gemini' in model:
        # 
        from roboreason.api_models import api_models
        rewards = []
        reasoning_traces = []
        for video_idx in range(len(downsampled_videos)):
            rewards_video_i, reasoning_traces_video_i = api_models(model, downsampled_videos[video_idx], task_description, key)
            rewards.append(rewards_video_i)
            reasoning_traces.append(reasoning_traces_video_i)
        # 
    elif model in ['sole-r1']: 
        from roboreason.sole import sole
        # from sole import load_model
        # load_model()
        rewards, reasoning_traces = sole(downsampled_videos, task_description, view_type_per_video=view_type_per_video, context_window=['current', 'previous', 'first'], model_path=model_path)
        # >>> rewards
        # [[0, 1, 0, 14, 16, 12, 7, 2, 0, -6]]
        # >>> reasoning_traces
        # [['', '<think>At the previous timestep, the gripper hovered over the table with the cube a short distance ahead and no contact made. At the current timestep, the gripper has moved slightly closer toward the cube and appears better aligned, but it still has not touched or grasped the cube. This small approach indicates incremental progress relative to the previous timestep. Given a previous progress of 0% and this minor closing of the gap, the current progress is predicted to be about 1%. To complete the task, the robot must continue approaching, align over the cube, close the gripper to secure it, and lift it off the table.</think><answer>1%</answer>', '<think>At both the previous and current timestep, the gripper has not picked up or made contact with the cube. The current timestep shows that the gripper has moved further from the cube compared to the previous timestep. Therefore, the task progress appears to be decreasing. Given that the previous task progress was 1%, the current task progress seems to have decreased to 0%.</think><answer>0%</answer>', '<think>At both the previous and current timestep, the gripper has not picked up or made contact with the cube. The current timestep shows that the gripper has moved closer to the cube compared to the previous timestep. Therefore, the task progress appears to be increasing. Given that the previous task progress was 0%, the current task progress seems to have increased to 14%.</think><answer>14%</answer>', '<think>At both the previous and current timestep, the gripper has not picked up or made contact with the cube. The current timestep shows that the gripper has moved closer to the cube compared to the previous timestep. Therefore, the task progress appears to be increasing. Given that the previous task progress was 14%, the current task progress seems to have increased to 16%.</think><answer>16%</answer>', '<think>At the previous timestep, the gripper was hovering near the cube on the table and appeared roughly aligned to approach it. At the current timestep, the gripper has shifted away from the cube and is not making contact, so it is farther from initiating a grasp than before. Because the end-effector moved away rather than closing in, progress likely decreased; given the previous progress of 16%, I would predict the current progress to be about 12%. To complete the task, the robot still needs to move closer to the cube, align the gripper around it, close the fingers to secure the cube, and lift it from the table.</think><answer>12%</answer>', '<think>At both the previous and current timestep, the gripper has not picked up or made contact with the cube. The current timestep shows that the gripper has moved further from the cube compared to the previous timestep. Therefore, the task progress appears to be decreasing. Given that the previous task progress was 12%, the current task progress seems to have decreased to 7%.</think><answer>7%</answer>', '<think>In the previous timestep, the cube sits on the table and the gripper is nearer to it, roughly aligned but not in contact. In the current timestep, the gripper has moved farther from the cube and remains open, so no grasp has been initiated and the cube is still resting on the table. Because the end-effector has increased its distance from the target, progress toward picking up the cube is decreasing relative to the previous timestep. Given a previous progress of 7%, this regression suggests the current progress is around 2%. To complete the task, the robot must re-approach the cube, align the gripper over it, close to grasp, and lift it off the table.</think><answer>2%</answer>', '<think>At the previous timestep, the gripper was hovering over the table and relatively near the cube but had not made contact. At the current timestep, the gripper has rotated and shifted slightly farther from the cube, still open and empty, indicating it moved away rather than toward a grasp. This suggests regression in the attempt to pick up the cube; given the previous progress of 2%, I would predict the current progress to be around 0%. To complete the task, the robot needs to approach the cube, align the gripper above it, close to grasp it, and lift it off the table.</think><answer>0%</answer>', '<think>At the previous timestep, the cube sits on the table while the gripper hovers nearby without contact. At the current timestep, the gripper has shifted slightly upward and away from the cube, increasing the distance and still showing no grasp. This motion moves the robot away from initiating a pickup, so given the previous progress of 0%, I would predict the current progress has decreased slightly to about -6%. To complete the task, the gripper needs to move back toward the cube, align over it, close to secure the cube, and lift it from the table.</think><answer>-6%</answer>']]
        # >>> rewards
        # [[0, 2, 3, 15, 17, 16, 9, 2, -1, -7]]
        # >>> reasoning_traces
        # [['', "<think>At the previous timestep, the gripper was hovering farther from the red cube with a noticeable gap and no contact. At the current timestep, the arm has moved closer and slightly lower toward the cube, while the cube remains on the table and the gripper still hasn't touched it. Given the previous progress of 0%, this small approach indicates increasing progress; I would predict the current progress to be about 2%. To finish the task, the robot must align over the cube, close the gripper to grasp it, and lift it off the table.</think><answer>2%</answer>", '<think>At both the previous and current timestep, the gripper has not picked up or made contact with the cube. The current timestep shows that the gripper has moved closer to the cube compared to the previous timestep. Therefore, the task progress appears to be increasing. Given that the previous task progress was 2%, the current task progress seems to have increased to 3%.</think><answer>3%</answer>', '<think>At the previous timestep, the gripper was hovering above the table with the cube clearly visible below and some lateral offset from it. At the current timestep, the gripper has moved closer toward the cube and appears better aligned above it, but it is still open and has not made contact or lifted the cube. This approach indicates increasing progress relative to the previous timestep; given the earlier estimate of 3%, the observed closing of the distance and improved alignment suggests the current progress is likely around 15%. To complete the task, the robot still needs to finish positioning directly over the cube, descend to an appropriate grasp pose, close the gripper to secure the cube, and then lift it from the table.</think><answer>15%</answer>', '<think>At both the previous and current timestep, the gripper has not picked up or made contact with the cube. The current timestep shows that the gripper has moved closer to the cube compared to the previous timestep. Therefore, the task progress appears to be increasing. Given that the previous task progress was 15%, the current task progress seems to have increased to 17%.</think><answer>17%</answer>', '<think>In the previous timestep, the gripper is hovering above the table with the cube clearly on the surface, and there is no contact. In the current timestep, the gripper has moved slightly farther away from the cube and remains open, so it is less aligned for a grasp than before. Given that the previous task progress was 17%, this increased distance suggests a small setback; I would predict the current progress to be about 16%. To complete the task, the robot still needs to re-approach the cube, align the gripper around it, close to secure a grasp, and then lift it from the table.</think><answer>16%</answer>', '<think>At the previous timestep, the gripper hovered over the table near the cube with the fingers open and no contact. At the current timestep, the arm has rotated and shifted slightly away from the cube, increasing the distance and still not making contact. Because the end effector moved farther from the target, progress toward picking up the cube is likely decreasing; given a previous progress of 16%, I would predict the current progress to be around 9%. To complete the task, the robot needs to re-approach the cube, align above it, close the gripper to secure it, and lift it off the table.</think><answer>9%</answer>', '<think>In the previous timestep, the gripper was positioned closer to the cube on the table and roughly aligned for an approach, but it had not made contact. In the current timestep, the gripper has rotated and moved farther from the cube, increasing the gap and still showing no grasp. Because the end-effector is retreating rather than closing in, the task appears to be regressing; given a previous progress of 9%, I would predict the current progress to be around 2%. To complete the task, the robot needs to re-approach the cube, align the gripper above it, descend, close the fingers to secure the cube, and lift it off the table.</think><answer>2%</answer>', '<think>At both the previous and current timestep, the gripper has not picked up or made contact with the cube. The current timestep shows that the gripper has moved further from the cube compared to the previous timestep. Therefore, the task progress appears to be decreasing. Given that the previous task progress was 2%, the current task progress seems to have decreased to -1%.</think><answer>-1%</answer>', '<think>At the previous timestep, the open gripper hovered near the cube on the table without making contact. At the current timestep, the arm has shifted slightly away from the cube, increasing the distance, and the gripper remains open with no contact. Because the end-effector is moving farther from the target instead of approaching or grasping it, progress is decreasing; given the previous progress of -1%, I would predict the current progress to be around -7%. To complete the task, the robot needs to move back toward the cube, align the gripper around it, close to secure a grasp, and lift it off the table.</think><answer>-7%</answer>']]
    # 
    elif model in ['topreward']:
        from roboreason.topreward import topreward
        rewards = []
        for video_idx in range(len(downsampled_videos)):
            rewards_video_i = topreward(downsampled_videos[video_idx], task_description, model_path=model_path)
            rewards.append(rewards_video_i)
    # 
    elif model in ['roboreward']:
        from roboreason.roboreward import roboreward, RoboRewardModel
        rewards = []
        for video_idx in range(len(downsampled_videos)):
            if model_path is None:
                rewards_video_i = roboreward(downsampled_videos[video_idx], task_description)
            else:
                rewards_video_i = roboreward(downsampled_videos[video_idx], task_description, model=RoboRewardModel(model_name=model_path))
            rewards.append(rewards_video_i)
    # 
    elif model in ['robometer']:
        from roboreason.robometer.roboreason_robometer import robometer
        # 
        rewards = []
        success_probs = []
        # for video_path in video_paths:
        for video_idx in range(len(downsampled_videos)):
            # rewards_video_i, success_probs_video_i = robometer(video_path, task_description)
            rewards_video_i, success_probs_video_i = robometer(downsampled_videos[video_idx], task_description, model_path=model_path)
            rewards.append(rewards_video_i)
            success_probs.append(success_probs_video_i)
    # 
    else:
        raise ValueError(f'Unknown model: {model}')
    # 
    ############ INTERPOLATE REWARD AND REASONING TRACES TO ALL FRAMES IN THE VIDEO (e.g. via linear interpolation)
    full_rewards = []
    full_success_probs = []
    full_reasoning_traces = []
    for video_idx in range(len(videos)):
        if not len(downsample_idx_list_list[video_idx]) == len(rewards[video_idx]):
            assert False, f"Length of downsample_idx_list {len(downsample_idx_list_list[video_idx])} does not match length of valid_answer_list {len(rewards[video_idx])} for video_idx {video_idx}"
        # 
        if len(downsample_idx_list_list[video_idx]) != len(videos[video_idx]):
            target_indices = np.arange(len(videos[video_idx]))
            x = np.asarray(downsample_idx_list_list[video_idx], dtype=float)
            y = np.asarray(rewards[video_idx], dtype=float)
            interp_rewards = np.interp(target_indices, x, y)
            # [int(x) for x in interp_rewards.tolist()]
            # rewards[video_idx]
            full_rewards.append(interp_rewards.tolist())
            # 
            if success_probs is not None:
                y = np.asarray(success_probs[video_idx], dtype=float)
                interp_success_probs = np.interp(target_indices, x, y)
                full_success_probs.append(interp_success_probs.tolist())
            # 
            if reasoning_traces is not None:
                full_reasoning_traces_video_i = []
                for step_idx in target_indices:
                    # step_idx = 0
                    if step_idx in downsample_idx_list_list[video_idx]:
                        full_reasoning_traces_video_i.append(reasoning_traces[video_idx][downsample_idx_list_list[video_idx].index(step_idx)])
                    else:
                        full_reasoning_traces_video_i.append(" ")
                full_reasoning_traces.append(full_reasoning_traces_video_i)
            else:
                full_reasoning_traces = None
        else:
            full_rewards = rewards
            full_reasoning_traces = reasoning_traces
            full_success_probs = success_probs
    # 
    if full_reasoning_traces is not None:
        if not single_video:
            return full_rewards, full_reasoning_traces
        else:
            return full_rewards[0], full_reasoning_traces[0]
    elif success_probs is not None:
        if not single_video:
            return full_rewards, full_success_probs
        else:
            return full_rewards[0], full_success_probs[0]
    else:
        if not single_video:
            return full_rewards
        else:
            return full_rewards[0]


def shape_to_target(frame, target=384):
    frame_height, frame_width = frame.shape[:2]
    if frame_height > target:
        frame = cv2.resize(frame, (frame_width*target//frame_height, target), interpolation=cv2.INTER_AREA)
    if frame_height < target: 
        frame = cv2.copyMakeBorder(frame, (target-frame_height)//2, (target-frame_height)//2, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    if frame_width < target:
        frame = cv2.copyMakeBorder(frame, 0, 0, (target-frame_width)//2, (target-frame_width)//2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return frame

def video_plot(outputs, plot_save_path, video_path=None, video_view_external_path=None, video_view_wrist_path=None, video_frames=None, view_type=None, show_all_frames=False,
               fps_=2, wrap_width=26, font_scale=1, font_height=30, text_thickness=2, line_type=2, show_reasoning_traces=True, cfg=None, env_rew_lab='Ground-truth reward'):
    # 
    ############ EXTRACT VIDEO FRAMES FOR ALL VIDEOS AS A LIST OF LISTS    
    # 
    if video_frames is None:
        if video_path is None and video_view_external_path is not None and video_view_wrist_path is not None:
            frames_external = load_video_frames(video_view_external_path[video_idx])
            frames_wrist = load_video_frames(video_view_wrist_path[video_idx])
            if not len(frames_external) == len(frames_wrist):
                raise ValueError(f"Number of frames in external view video {len(frames_external)} does not match number of frames in wrist view video {len(frames_wrist)} for video_idx {video_idx}")
            frame_list = [np.concatenate((frames_external[i], frames_wrist[i]), axis=1) for i in range(len(frames_external))]
            view_type = 'external and wrist'
        else:
            if video_path is None and video_view_external_path is not None and video_view_wrist_path is None:
                video_path = video_view_external_path
                view_type = 'external'
                print(f"Using videos from video_view_external_path with view type 'external'")
            elif video_path is None and video_view_external_path is None and video_view_wrist_path is not None:
                video_path = video_view_wrist_path
                view_type = 'wrist'
                print(f"Using videos from video_view_wrist_path with view type 'wrist'")
            elif video_path is None:
                raise ValueError("video_path cannot be None if video_view_external_path and video_view_wrist_path are both not provided (and video_frames is also None)")
            # 
            frame_list = load_video_frames(video_path)
    else:
        frame_list = video_frames
    #   
    # only show reasoning traces in video if only showing one model - if "reasoning_traces" in more than 2 outputs, set reasoning_traces to None since this is too much information to show in the video
    count_reasoning_traces = 0
    output_rewards_len = []
    output_rewards_concat = []
    final_reasoning_traces_idx_list = []
    for output in outputs:
        output_rewards_len.append(len(output["rewards"]))
        output_rewards_concat += output["rewards"]
        if "reasoning_traces" in output and output["reasoning_traces"] is not None:
            count_reasoning_traces += 1
            final_reasoning_traces = output["reasoning_traces"]
            final_reasoning_traces_idx_list = []
            for step_idx in range(len(final_reasoning_traces)):
                if step_idx==0 or not final_reasoning_traces[step_idx].strip() == "":
                    final_reasoning_traces_idx_list.append(step_idx)
    # 
    if not count_reasoning_traces == 1:
        show_reasoning_traces = False
    # 
    if show_reasoning_traces:
        if not show_all_frames:
            print('Only showing reasoning traces for frames where reasoning traces are not empty. If you want to show reasoning traces for all frames, set show_all_frames to True.')
            output_rewards_len = []
            output_rewards_concat = []
            for output_idx in range(len(outputs)):
                outputs[output_idx]['reasoning_traces'] = [outputs[output_idx]['reasoning_traces'][i] for i in final_reasoning_traces_idx_list]
                outputs[output_idx]['rewards'] = [outputs[output_idx]['rewards'][i] for i in final_reasoning_traces_idx_list]
                output_rewards_len.append(len(outputs[output_idx]['rewards']))
                output_rewards_concat += outputs[output_idx]['rewards']
            final_reasoning_traces = [final_reasoning_traces[i] for i in final_reasoning_traces_idx_list]
            frame_list = [frame_list[i] for i in final_reasoning_traces_idx_list]
    # 
    text_title=''
    if True:
        plt.rcParams.update({'font.size': 12})  # Adjust font size as needed
        #             # 
        frame = frame_list[0]
        # if isinstance(frame, Image.Image):
        #     frame = np.array(frame.convert("RGB"))
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #
        target_height = 384*2
        frame = shape_to_target(frame, target=target_height)
        frame_height, frame_width = frame.shape[:2]
        # 
        # if view_type in ['external and wrist']:
        #     if show_reasoning_traces:
        #         output_width = target_height*4
        #     else:
        #         output_width = target_height*3
        # else:
        #     if show_reasoning_traces:
        #         output_width = target_height*3
        #     else:
        #         output_width = target_height*2
        if show_reasoning_traces:
            output_width = frame_width + (target_height*2)
        else:
            # output_width = target_height*3
            output_width = frame_width + target_height
        # 
        output_height = frame_height
        # 
        print('output_width, output_height:', output_width, output_height)
        if plot_save_path.endswith(".mp4"):
            plot_save_path = plot_save_path.replace(".mp4", ".avi")
        # 
        if not os.path.exists(os.path.dirname(plot_save_path)):
            os.makedirs(os.path.dirname(plot_save_path))
        # 
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(plot_save_path, fourcc, fps_, (frame_width, frame_height))
        out = cv2.VideoWriter(plot_save_path, fourcc, fps_, (output_width, output_height))
        # 
        fig, ax = plt.subplots(dpi=200)  # Increase DPI for higher resolution
        canvas = FigureCanvas(fig)
        # xdata, ydata, ydata2 = [], [], []
        xdata, ydata, ydata2, ydata3, ydata4, ydata5, ydata6, ydata7 = [], [], [], [], [], [], [], []
        # xlim_max = max(len(predicted_rewards), len(groundtruth_rewards))
        xlim_max = max(output_rewards_len)
        ax.set_xlim(0, xlim_max-1)
        ax.set_title(text_title)
        ymax=100
        # 
        # ax.set_ylim(min(predicted_rewards+groundtruth_rewards), 100)
        ax.set_ylim(min(output_rewards_concat), 100)
        ax.set_ylabel('Predicted task progress (%)')
        ax.set_xlabel('Step number')
        # 
        ln, = ax.plot([], [], 'r', linewidth=5,label=outputs[0]["model"])
        ln.set_color('darkblue')
        ln.set_markerfacecolor('darkblue')
        ln.set_markersize(5)
        # 
        ln2, ln3, ln4, ln5, ln6, ln7 = None, None, None, None, None, None
        if len(outputs) > 1:
            ln2, = ax.plot([], [], 'b', linewidth=5,label=outputs[1]["model"])
            ln2.set_color('darkred')
            ln2.set_markerfacecolor('darkred')
            ln2.set_markersize(5)
        if len(outputs) > 2:
            ln3, = ax.plot([], [], 'g', linewidth=5,label=outputs[2]["model"])
            ln3.set_color('darkgreen')
            ln3.set_markerfacecolor('darkgreen')
            ln3.set_markersize(5)
        if len(outputs) > 3:
            ln4, = ax.plot([], [], 'y', linewidth=5,label=outputs[3]["model"])
            ln4.set_color('orange')
            ln4.set_markerfacecolor('orange')
            ln4.set_markersize(5)
        if len(outputs) > 4:
            ln5, = ax.plot([], [], 'c', linewidth=5,label=outputs[4]["model"])
            ln5.set_color('cyan')
            ln5.set_markerfacecolor('cyan')
            ln5.set_markersize(5)
        if len(outputs) > 5:
            ln6, = ax.plot([], [], 'm', linewidth=5,label=outputs[5]["model"])
            ln6.set_color('magenta')
            ln6.set_markerfacecolor('magenta')
            ln6.set_markersize(5)
        if len(outputs) > 6:
            ln7, = ax.plot([], [], 'k', linewidth=5,label=outputs[6]["model"])
            ln7.set_color('black')
            ln7.set_markerfacecolor('black')
            ln7.set_markersize(5)
        if len(outputs) > 7:
            assert False, f"Plotting more than 7 outputs in the same video is not currently supported."
        # 
        ax.legend(fontsize='small')
        frame_number = 0
        data_point_number = 0
        while frame_number<xlim_max:
            frame = frame_list[frame_number]
            # if isinstance(frame, Image.Image):
            #     frame = np.array(frame.convert("RGB"))
            #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 
            frame = shape_to_target(frame, target=target_height)
            frame_height, frame_width = frame.shape[:2]
            fig.set_size_inches(target_height / fig.dpi, target_height / fig.dpi)  
            fig.tight_layout(pad=0.4)  # Adjust layout to ensure everything fits without being cut off
            # 
            if True:
                xdata.append(data_point_number)
                # if len(predicted_rewards) > data_point_number:
                if len(outputs[0]["rewards"]) > data_point_number:
                    ydata.append(outputs[0]["rewards"][data_point_number])
                    ln.set_data(xdata, ydata)
                    ax.draw_artist(ln)
                # 
                ax.draw_artist(ax.patch)
                # if len(groundtruth_rewards) > data_point_number:
                if ln2 is not None:
                    ydata2.append(outputs[1]["rewards"][data_point_number])
                    ln2.set_data(xdata, ydata2)
                    ax.draw_artist(ln2)
                if ln3 is not None:
                    ydata3.append(outputs[2]["rewards"][data_point_number])
                    ln3.set_data(xdata, ydata3)
                    ax.draw_artist(ln3)
                if ln4 is not None:
                    ydata4.append(outputs[3]["rewards"][data_point_number])
                    ln4.set_data(xdata, ydata4)
                    ax.draw_artist(ln4)
                if ln5 is not None:
                    ydata5.append(outputs[4]["rewards"][data_point_number])
                    ln5.set_data(xdata, ydata5)
                    ax.draw_artist(ln5)
                if ln6 is not None:
                    ydata6.append(outputs[5]["rewards"][data_point_number])
                    ln6.set_data(xdata, ydata6)
                    ax.draw_artist(ln6)
                if ln7 is not None:    
                    ydata7.append(outputs[6]["rewards"][data_point_number])
                    ln7.set_data(xdata, ydata7)
                    ax.draw_artist(ln7)
                # 
                canvas.draw()
                plot_image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
                plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (4,))
            # 
            # 
            plot_image_resized = cv2.resize(plot_image, (target_height, target_height), interpolation=cv2.INTER_LINEAR)
            # plot_image_rgb = cv2.cvtColor(plot_image_resized, cv2.COLOR_RGBA2RGB)
            plot_image_bgr = cv2.cvtColor(plot_image_resized, cv2.COLOR_RGBA2BGR)
            # 
            if show_reasoning_traces:
                # white_box_width = frame_width // denom_  # Adjust width of the white box
                white_box_width = target_height  # Set white box width to match the height of the frame
                white_box = np.ones((frame_height, white_box_width, 3), dtype=np.uint8) * 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_color = (0, 0, 0)  
                thickness = text_thickness
                line_type = line_type
                text = final_reasoning_traces[frame_number]
                language_description_split = text.split(' ')
                language_description_wrapped = ''
                line_i_len = 0
                for i in range(len(language_description_split)):
                    if line_i_len < wrap_width:
                        language_description_wrapped += language_description_split[i] + ' '
                        line_i_len += len(language_description_split[i])
                    else:
                        language_description_wrapped += '\n' + language_description_split[i] + ' '
                        line_i_len = len(language_description_split[i])
                text = language_description_wrapped
                text = text.replace('</think><answer>', '</think>\n<answer>')
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (white_box_width - text_size[0]) // 2
                text_y = (frame_height + text_size[1]) // 2
                lines = text.split('\n')
                start_y = font_height 
                for i, line in enumerate(lines):
                    y = start_y + (i * font_height)  
                    res_ = cv2.putText(white_box, line, (10, y), font, font_scale, font_color, thickness, line_type)
                # 
                # combined_frame = np.hstack((frame, white_box, plot_image_rgb))
                combined_frame = np.hstack((frame, white_box, plot_image_bgr))
            else:
                # combined_frame = np.hstack((frame, plot_image_rgb))
                combined_frame = np.hstack((frame, plot_image_bgr))
            # 
            combined_frame = combined_frame.astype(np.uint8)
            assert combined_frame.shape[0] == output_height
            assert combined_frame.shape[1] == output_width
            assert combined_frame.dtype == np.uint8
            # out.write(combined_frame)
            out.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
            frame_number += 1
            data_point_number += 1
        # 
        out.release()
    plt.close(fig)   
    del canvas 
    os.system(f"ffmpeg -y -i {plot_save_path} -c:v libx264 -pix_fmt yuv420p {plot_save_path.replace('.avi', '.mp4')}")
    os.remove(plot_save_path)



