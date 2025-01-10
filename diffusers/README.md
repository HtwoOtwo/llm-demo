# 在单张4090上训练diffusers

## 微调Flux
"NOTE"部分是需要注意的，如果想使用24G显存训练Flux，量化是必要的，而且最好是在cpu上量化好再move到gpu上。

## 微调Inpainting

### 准备数据
运行 inpainting_label.py, 开始标注
标注完成后会在指定的save_folder下生成数据集:

```
save_folder
   - image1_name
        - raw_image.ext
        - prompt1.png (mask)
        - prompt2.png (mask)
        - ...
    - image2_name
        - raw_image.ext
        - prompt1.png (mask)
        - prompt2.png (mask)
        - ...
```

### 运行训练脚本
train_dreambooth_lora_inpainting.py

data_root结构为
```
save_folder
   - image1_name
        - raw_image.ext
        - prompt1.png (mask)
        - prompt2.png (mask)
        - ...
    - image2_name
        - raw_image.ext
        - prompt1.png (mask)
        - prompt2.png (mask)
        - ...
```