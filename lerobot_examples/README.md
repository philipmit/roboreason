
---
## Examples of annotating lerobot datasets 
```python
# pip install -U roboreason
import roboreason as rr

from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset_name = "jackvial/so101_pickplace_recap_merged_v2"
dataset = LeRobotDataset(dataset_name, video_backend="pyav")

task_description="Pick up the cube from the table and place it at the target location marked by the X."

# NOTE: SOLE-R1 is fastest, but all methods are slow with large datasets

# uses lerobot add_features function to add reward annotations to the dataset, stored in the with_reward sub-directory
rr.annotate(model="robometer", lerobot_dataset=dataset, observation_name="observation.images.side", annotation_version="v1", task_description=task_description, num_reasoning_frames=10)
rr.annotate(model="sole-r1", lerobot_dataset=dataset, observation_name="observation.images.side", annotation_version="v1", task_description=task_description,  num_reasoning_frames=10)
rr.annotate(model="roboreward", lerobot_dataset=dataset, observation_name="observation.images.side", annotation_version="v1", task_description=task_description, num_reasoning_frames=10)
rr.annotate(model="topreward", lerobot_dataset=dataset, observation_name="observation.images.side", annotation_version="v1", task_description=task_description, num_reasoning_frames=10)
```


## Extracting the saved annotations and visualizing
```python
video_frames = rr.extract_frames(lerobot_dataset=dataset, observation_name="observation.images.side")
rewards_robometer = rr.extract_annotation(lerobot_dataset=dataset, model="robometer", annotation_version="v1")
rewards_sole = rr.extract_annotation(lerobot_dataset=dataset, model="sole-r1", annotation_version="v1")
rewards_roboreward = rr.extract_annotation(lerobot_dataset=dataset, model="roboreward", annotation_version="v1")
rewards_topreward = rr.extract_annotation(lerobot_dataset=dataset, model="topreward", annotation_version="v1")


plot_save_name_base = dataset_name.replace('/', '_')
for video_idx in range(len(video_frames)):
    output_sole = {"model": "sole-r1", "rewards": rewards_sole[video_idx]}
    plot_save_dir = f"../model_outputs/sole-r1/lerobot/{plot_save_name_base}/"
    rr.video_plot(outputs=[output_sole], plot_save_path=plot_save_dir + f"{plot_save_name_base}_episode_{video_idx}.mp4", video_frames=video_frames[video_idx], view_type='external', show_reasoning_traces=False, show_all_frames=False, fps_=10)
    # 
    output_robometer = {"model": "robometer", "rewards": rewards_robometer[video_idx]}
    plot_save_dir = f"../model_outputs/robometer/lerobot/{plot_save_name_base}/"
    rr.video_plot(outputs=[output_robometer], plot_save_path=plot_save_dir + f"{plot_save_name_base}_episode_{video_idx}.mp4", video_frames=video_frames[video_idx], view_type='external', show_reasoning_traces=False, show_all_frames=False, fps_=10)
    # 
    output_roboreward = {"model": "roboreward", "rewards": rewards_roboreward[video_idx]}
    plot_save_dir = f"../model_outputs/roboreward/lerobot/{plot_save_name_base}/"
    rr.video_plot(outputs=[output_roboreward], plot_save_path=plot_save_dir + f"{plot_save_name_base}_episode_{video_idx}.mp4", video_frames=video_frames[video_idx], view_type='external', show_reasoning_traces=False, show_all_frames=False, fps_=10)
    # 
    output_topreward = {"model": "topreward", "rewards": rewards_topreward[video_idx]}
    plot_save_dir = f"../model_outputs/topreward/lerobot/{plot_save_name_base}/"
    rr.video_plot(outputs=[output_topreward], plot_save_path=plot_save_dir + f"{plot_save_name_base}_episode_{video_idx}.mp4", video_frames=video_frames[video_idx], view_type='external', show_reasoning_traces=False, show_all_frames=False, fps_=10)
    # 
    output_combined = [output_robometer, output_sole, output_roboreward, output_topreward]
    plot_save_dir = f"../model_outputs/combined/lerobot/{plot_save_name_base}/"
    rr.video_plot(outputs=output_combined, plot_save_path=plot_save_dir + f"{plot_save_name_base}_episode_{video_idx}.mp4", video_frames=video_frames[video_idx], view_type='external', show_reasoning_traces=False, show_all_frames=False, fps_=10)

```







