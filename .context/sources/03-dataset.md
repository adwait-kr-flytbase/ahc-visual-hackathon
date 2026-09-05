# **AHC Visual Intelligence Hackathon — Training and Public Test Data**

**Download Training and test dataset (mirror links):**

**Mirror 1:  [Train and Test](https://drive.google.com/drive/folders/1sEFKR7ctd5GfFw-nMlYd_MnTw1VVYz9K?usp=sharing)**

**Mirror 2:  [Train and Test - Mirror 2](https://drive.google.com/drive/folders/13E_CePn14lcbwMA_yZEiHpAVx6i09UIG?usp=sharing)**

**Mirror 3:  [Train and Test - Mirror 3](https://drive.google.com/drive/folders/13V8JqgZRMzn2TCF0HTsCqVgUH0UOMmpb?usp=sharing)**

**Mirror 4:  [Train and Test - Mirror 4](https://drive.google.com/drive/folders/1fS_i7QKXRDI6mnaI6UWqYzKSOYWG8rFv?usp=sharing)**

**Mirror 5:  [Train and Test - Mirror 5](https://drive.google.com/drive/folders/1efhUZhB6Kyvpw3RulZJSwd0brb8KhuZf?usp=sharing)**

This dataset supports the AHC near-real-time video anomaly detection hackathon. It contains curated video data for developing systems that identify safety and infrastructure anomalies in CCTV, dashcam and drone footage.

## **Dataset contents**

| train/  \<class\_name\>/    videos/\*.mp4    videos.csv    ground\_truth.csvtest/  videos/\*.mp4  videos.csv  ground\_truth.csv |
| :---- |

## **Training data**

## The training videos are real samples selected from the available source datasets. They include short event clips, normal contextual clips and longer temporal videos. Each final label has its own folder, including `normal`. The same media can be used in three ways:

1. ## As raw video input for teams building their own preprocessing or sampling pipeline.

2. ## With the class folder's `ground_truth.csv` for anomaly, class and temporal supervision.

3. ## With `description_summary` in that CSV for vision-language fine-tuning or distillation.

## The complete train-and-test pack is approximately 15–17 GB. Videos are standardized to a practical delivery format while preserving the original visual events. No synthetic anomaly footage is included.

## 

## 

## **Public test data**

## The public test set contains 34 videos totaling approximately 56 minutes. `ground_truth.csv` is included so teams can validate their output and scoring pipeline before submitting to the private evaluation system.

## **Camera domains and environments**

## The data covers:

* ## CCTV, dashcam and drone viewpoints

* ## Highways, city streets, intersections, campuses and open areas

* ## Day, night and difficult weather or visibility conditions

* ## Short isolated events and longer multi-event temporal videos

## **Label set**

1. ## `normal`

2. ## `traffic_accident`

3. ## `traffic_congestion`

4. ## `stalled_or_broken_down_vehicle`

5. ## `vehicle_blocking_traffic`

6. ## `wrong_way_driving`

7. ## `road_spill_or_debris`

8. ## `waterlogging_or_flood`

9. ## `fire`

10. ## `smoke`

11. ## `fighting_or_violence`

12. ## `loitering_or_suspicious_presence`

## 

## **Annotation files**

## Every training class folder has the same two-file metadata pattern as the public test set: `videos.csv` maps IDs to files and `ground_truth.csv` uses one row per event. Normal videos have one row with `class_name=normal` and empty event timestamps.

## Ground-truth fields are:

| Column | Notes |
| ----- | ----- |
| `video_id` | Repeats — one video can hold several events |
| `level` | 1, 2 or 3 — the task tier (below) |
| `is_anomaly` | The binary label |
| `class_name` | One of twelve strings — match exactly |
| `start_time_sec` / `end_time_sec` | Empty on Level 1, populated on Levels 2–3 |
| `description_summary` | Short natural-language description; sometimes blank |

### 

## `test/videos.csv` maps public test video IDs to their files. `test/ground_truth.csv` contains the public labels and event intervals.

## 

## **Data separation**

## Training sources are separated from both the public test set and private evaluation set at the original source-video or source-sequence level. Different cuts from a reserved benchmark source are not placed in training.

## **Intended use**

## The dataset is intended for hackathon development, model training, fine-tuning, distillation and local validation of near-real-time video anomaly detection systems. Teams remain responsible for following the usage terms of the underlying source datasets.

## 

