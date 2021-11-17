
## VPReID
video person re-identification   
个人毕设记录  代码在全部完成并整理后 上传  
  
## 环境安装  
参考 https://github.com/JDAI-CV/fast-reid  
```shell script
conda create -n fastreid python=3.7
conda activate fastreid
conda install pytorch==1.6.0 torchvision tensorboard -c pytorch
```
## TODO   

## 算法 
- 1.MST
  - GemMe (gemP motion enhance)
  - STB (spatial temporal block)
  - RFE (random frame feature eraseing)
 
- 2.MRA (Multi Range Aggregation)
  - RFG (range feature generate)
  - GRR (global range reference)
  - SFM (stepwise fusion module) 
 
##  baseline指标  
括号中表示split_id  

|      Datasets  (R1/mAP)    | baseline       |   
|      :----------------:    | :-----------:  | 
| prid                       |                | 
| ilids                      |                | 
|        Mars                |                |  
|       DukeV                |                |   
|       LSVID                |                |   
------------------------------------------------

**只记录真实的数据指标**  

|      Datasets  (R1/mAP)    | MST (2X1080TI)  |    MRA (2X1080TI)  |  MST-MRA (A100) |  
|      :----------------:    | :-----------:   |  :-----------:     |   :-----------: |  
| prid                       |     96.6/--(0)  | 94.4/--(0)         |                 |
| ilids                      |     95.3/--(0)  | 92.0/--(0)         |                 |  
|        Mars                |    91.2/86.3    | 91.1/85.5          | 91.6/85.6       |
|       DukeV                |    97.1/97.6    | 98.4/97.4          |                 | 
|       LSVID                |                 | 78.9/67.9          |                 | 
 

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
