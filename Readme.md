# MVP-Bench: Can Large Vision-Language Models Conduct Multi-level Visual Perception Like Humans?

This repository contains data and code for the paper [MVP-Bench: Can Large Vision-Language Models Conduct Multi-level Visual Perception Like Humans?](). 

## MVP-Bench

The benchmark is located in the data folder. `all_question.json` file contains all the visual questions, while `mcq_questions` contain all the multiple-choice questions under Circular Strategy within MVP-Bench. You can download the visual data from [here](https://huggingface.co/datasets/GZClarence/MVP-Bench). To get access to the MVP-Bench and avoid the misuse of the benchmark, please fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLScSQff_VLCDXyvsV5I_D_plDTgRQBrDFcpuhXJUDMZx-K17VQ/viewform).

As for images, the folder `data/Cross_Images` contains all the images for single-image tasks, while the folder `data/Single_Images` contains images for cross-image tasks.

## Evaluation

Our Evaluation is based on [the VLMEvalKit](https://github.com/open-compass/VLMEvalKit). The evaluation consists of two steps: Inference and Evaluation.

Here is an example of running inference with command:
```shell
python inference.py --model_name GPT4o --img_dir 'data/Images' --output_dir 'model_predictions' --qas_pth 'data/all_questions.json' --question_type 'all_questions'
```

Here is an example of running evaluation with command:
```shell
python evaluate.py --model_name GPT4o
```

Our experiment results have been stored in the `model_prediction` folder.