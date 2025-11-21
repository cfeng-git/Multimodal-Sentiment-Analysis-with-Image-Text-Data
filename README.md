# Multimodal Sentiment Analysis with Image-Text Data




## Writing the config file

The config file takes in the following

- `training`
    - `batch_size`
    - `num_epochs`
    - `learning_rate`
    - `weight_decay`
    - `seed`

- `model` 
    - `model_name`
    - `text_model`
    - `vision_model`
    - `max_length`

- `data`
    - `train_csv`
    - `val_csv`
    - `target`: the ground truth labels, could choose `text`, `image` or `combined`


- `logging`
    - `output_dir`
    - `save_every`: ? not implemented



## Dataset

[MVSA Multiview Dataset](./docs/MVSA_MV_dataset.md)

