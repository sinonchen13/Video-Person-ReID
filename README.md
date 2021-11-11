## VPReID
video person re-identification 

## 环境安装  
参考 https://github.com/JDAI-CV/fast-reid  
```shell script
conda create -n fastreid python=3.7
conda activate fastreid
conda install pytorch==1.6.0 torchvision tensorboard -c pytorch
```
## 算法 
- 1.GCM
  - GemMe
  - CoordAtt3D
  - MultiLoss 
- 2.MRA (Multi Range Aggregation)
  - Range Feature Gen
  - Global Reference Module
  - Stepwise Fusion Module
 
##  baseline指标  
括号中表示split_id  后续统一一下  

|      Datasets  (R1/mAP)    | baseline       |   
|      :----------------:    | :-----------:  | 
| prid                       |    94.4/-- (9) | 
| ilids                      |    82.7/-- (9) | 
|        Mars                |    88.9/83.4   |  
|       DukeV                |    95.9/95.3   |   
|       LSVID                |    81.5/72.1   |   
------------------------------------------------

**只记录真实的数据指标**  

|      Datasets  (R1/mAP)    | GCM             |    MRA         |
|      :----------------:    | :-----------:   |  :-----------: |
| prid                       |     92.1/--(9)  | 94.4/--(0)     |
| ilids                      |     86.0/--(9)  | 92.0/--(0)     |
|        Mars                |    90.2/85.4    | 91.1/85.5      |
|       DukeV                |                 | 98.4/97.4      |
|       LSVID                |                 | 78.9/67.9      |
 

------------------------------------------------- 
## 可视化Demo  
- gt demo  

  ![gt image](pic/0912C5T0006F001_gt.jpg) 

- result demo  

  ![res image](pic/0912C5T0006F001.jpg)  

## Mars Test Log

    Loading checkpoint from 'Mars_new_integrate0231/checkpoint_ep220.pth.tar'  
    Evaluate  
    1000/1980  
    Extracted features for query set, obtained torch.Size([1980, 2048]) matrix  
    1000/9330  
    2000/9330  
    3000/9330  
    4000/9330  
    5000/9330  
    6000/9330  
    7000/9330  
    8000/9330  
    9000/9330  
    Extracted features for gallery set, obtained torch.Size([11310, 2048]) matrix  
    Extracting features complete in 62m 48s  
    Computing distance matrix  
    Computing CMC and mAP  
    ori Results ----------  
    top1:90.2% top5:97.1% top10:98.0% mAP:85.4%  
    re Results ----------  
    top1:94.4% top5:98.0% top10:98.6% mAP:88.5%
