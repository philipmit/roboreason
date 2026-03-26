
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

# uses lerobot add_features function to add reward annotations to the dataset, stored in the '/with_reward' sub-directory
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


## Reading the saved rewards as pandas df
```python
import os
import pandas as pd
parquet_path = os.path.join(dataset.root / "with_reward" , "data/chunk-000/file-000.parquet")
df = pd.read_parquet(parquet_path)
md_table = df.head(10).to_markdown(index=False)
# | action                                                              | observation.state                                                   |   timestamp |   frame_index |   episode_index |   index |   task_index |   rewards_robometer_v1 |   rewards_sole-r1_v1 |   rewards_roboreward_v1 |   rewards_topreward_v1 |
# |:--------------------------------------------------------------------|:--------------------------------------------------------------------|------------:|--------------:|----------------:|--------:|-------------:|-----------------------:|---------------------:|------------------------:|-----------------------:|
# | [ 45.85219  -96.01361  100.        76.41357  -24.210526  38.436745] | [ 45.420906 -97.12963   97.00449   86.924034 -23.157345  38.10764 ] |   0         |             0 |               0 |       0 |            0 |                14.1052 |             0        |                       0 |                0       |
# | [ 45.85219  -96.01361  100.        76.41357  -24.210526  38.436745] | [ 45.420906 -97.12963   97.00449   86.924034 -23.157345  38.10764 ] |   0.0333333 |             1 |               0 |       1 |            0 |                14.292  |             0.205128 |                       0 |                2.4536  |
# | [ 45.85219  -96.01361  100.        76.41357  -24.210526  38.436745] | [ 45.420906 -96.94444   97.20419   86.176834 -23.575537  38.10764 ] |   0.0666667 |             2 |               0 |       2 |            0 |                14.4788 |             0.410256 |                       0 |                4.9072  |
# | [ 45.85219  -96.01361  100.        76.41357  -24.210526  38.436745] | [ 45.420906 -96.01852   97.60359   83.93524  -24.202824  38.10764 ] |   0.1       |             3 |               0 |       3 |            0 |                14.6657 |             0.615385 |                       0 |                7.3608  |
# | [ 45.85219  -96.01361  100.        76.41357  -24.210526  38.436745] | [ 45.420906 -95.833336  97.60359   80.69739  -24.202824  38.10764 ] |   0.133333  |             4 |               0 |       4 |            0 |                14.8525 |             0.820513 |                       0 |                9.81441 |
# | [ 45.85219  -96.01361  100.        76.41357  -24.210526  38.436745] | [ 45.420906 -96.57407   97.60359   79.20299  -24.202824  38.10764 ] |   0.166667  |             5 |               0 |       5 |            0 |                15.0394 |             1.02564  |                       0 |               12.268   |
# | [ 45.85219  -96.01361  100.        76.41357  -24.210526  38.436745] | [ 45.420906 -96.296295  97.60359   78.45579  -24.202824  38.10764 ] |   0.2       |             6 |               0 |       6 |            0 |                15.2262 |             1.23077  |                       0 |               14.7216  |
# | [ 45.85219  -96.01361  100.        76.41357  -24.210526  38.436745] | [ 45.420906 -95.74074   97.60359   77.95766  -24.202824  38.10764 ] |   0.233333  |             7 |               0 |       7 |            0 |                15.413  |             1.4359   |                       0 |               17.1752  |
# | [ 45.85219  -96.01361  100.        76.41357  -24.210526  38.436745] | [ 45.420906 -95.64815   97.60359   77.58406  -24.202824  38.10764 ] |   0.266667  |             8 |               0 |       8 |            0 |                15.5999 |             1.64103  |                       0 |               19.6288  |
# | [ 45.85219  -96.01361  100.        76.41357  -24.210526  38.436745] | [ 45.420906 -95.92593   97.60359   77.33499  -24.202824  38.10764 ] |   0.3       |             9 |               0 |       9 |            0 |                15.7867 |             1.84615  |                       0 |               22.0824  |

```




