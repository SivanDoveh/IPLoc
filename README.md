# Teaching VLMs to Localize Specific Objects from In-context Examples (IPLoc)
![teaser.pdf](https://github.com/SivanDoveh/ICLoc/blob/main/images/teaser.jpg)

## Environment Setup:
Prepare the Qwen2-VL environment as shown in [Qwen2VL Env](https://github.com/QwenLM/Qwen2-VL?tab=readme-ov-file#quickstart)

## Data Preparation:
1. Download and place the images in the respective folders:
   - per_seg: Download [PerSeg](https://paperswithcode.com/dataset/perseg) images 
   - LASOT: Download [LASOT](http://vision.cs.stonybrook.edu/~lasot/download.html) images 
   - frames(PDM): Download [PDM](https://github.com/dvirsamuel/PDM?tab=readme-ov-file#personalized-diffusion-features-matching-pdm) images

2. The folder structure should look like this:

   ```SIVAN
   data/
   ├── ICL_tracking/
       └── video/
           ├── frames/
           ├── per_seg/
           └── LASOT/
   ```

## Model Download:
Download our model from [QWEN2-VL-ICL-LOC](https://drive.google.com/drive/folders/1u_1Mj_WMqMhA51MzN8j1FugU0Z2p6RpA?usp=sharing)

## Evaluation:
To evaluate the ICLoc model, use the following command:
  ```SIVAN
python Loc_Qwen2VL7B.py --data_path ./Loc/data/test_data_path.json --name IPLocEval --lora_weights_path lora_pth_to_model
 ```

Test data JSON files are in the data directory, including ICL - PDM, LASOT, and PerSeg.





