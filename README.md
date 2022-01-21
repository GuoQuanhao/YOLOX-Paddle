# YOLOX-Paddle
A reproduction of YOLOX by PaddlePaddle

## 数据集准备
下载COCO数据集，准备为如下路径
```
/home/aistudio
|-- COCO
|   |-- annotions
|   |-- train2017
|   |-- val2017
```
除了常用的图像处理库，需要安装额外的包
```
pip install gputil==1.4.0 loguru pycocotools
```
进入仓库根目录，编译安装（**推荐使用AIStudio**）
```
cd YOLOX-Paddle
pip install -v -e .
```
如果使用本地机器出现编译失败，需要修改`YOLOX-Paddle/yolox/layers/csrc/cocoeval/cocoeval.h`中导入`pybind11`的include文件为本机目录，使用如下命令获取`pybind11`的`include`目录
```
>>> import pybind11
>>> pybind11.get_include()
'/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/pybind11/include'
```
如`AIStudio`路径
```
#include </opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/pybind11/include/pybind11/numpy.h>
#include </opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/pybind11/include/pybind11/pybind11.h>
#include </opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/pybind11/include/pybind11/stl.h>
#include </opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/pybind11/include/pybind11/stl_bind.h>
```
成功后使用`pip list`可看到安装模块
```
yolox    0.1.0    /home/aistudio/YOLOX-Paddle
```
设置`YOLOX_DATADIR`环境变量\或者\`ln -s /path/to/your/COCO ./datasets/COCO`来指定COCO数据集位置
```
export YOLOX_DATADIR=/home/aistudio/
```

## 训练
```
python tools/train.py -n yolox-nano -d 1 -b 64
```
得到的权重保存至`./YOLOX_outputs/nano/yolox_nano.pdparams`

## 验证
```
python tools/eval.py -n yolox-nano -c ./YOLOX_outputs/nano/yolox_nano.pdparams -b 64 -d 1 --conf 0.001
```
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.259
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.269
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.083
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.242
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.384
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.419
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.632
```
并提供了[官方预训练权重](https://pan.baidu.com/s/1N4l1A2YAA2etPqh1Lcg-yA)，code:ybxc
|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)|
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |40.5 |40.5      |9.8      |9.0 | 26.8 |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.9 |47.2      |12.3     |25.3 |73.8|
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |49.7 |50.1      |14.5     |54.2| 155.6 |
|[YOLOX-x](./exps/default/yolox_x.py)   |640   |51.1 |**51.5**  | 17.3    |99.1 |281.9 |
|[YOLOX-Darknet53](./exps/default/yolov3.py)   |640  | 47.7 | 48.0 | 11.1 |63.7 | 185.3 |

#### Light Models.

|Model |size |mAP<sup>val<br>0.5:0.95 | Params<br>(M) |FLOPs<br>(G)| weights | log |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: | :---: |
|[YOLOX-Nano](./exps/default/nano.py) |416  |25.9  | 0.91 |1.08 | [baidu](https://pan.baidu.com/s/1N4l1A2YAA2etPqh1Lcg-yA) code:ybxc | [Nano-log](./YOLOX_outputs/nano/train_log.txt)|
|[YOLOX-Tiny](./exps/default/yolox_tiny.py) |416  |32.9 | 5.06 |6.45 | [baidu](https://pan.baidu.com/s/1N4l1A2YAA2etPqh1Lcg-yA) code:ybxc |[Tiny-log](./YOLOX_outputs/yolox_tiny/train_log.txt) |
## 推理
```
python tools/demo.py image -n yolox-nano -c ./YOLOX_outputs/nano/yolox_nano.pdparams --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result
```
推理结果如下所示
 
<img src="https://ai-studio-static-online.cdn.bcebos.com/c199d241704b44469a2846967f4f1057f8f208d5c103417ca7c0c8590ecb084d" width="400"/><img src="https://ai-studio-static-online.cdn.bcebos.com/f7f8922c525040f08a8a52837948826911362f3f6b01406eb49945dd6925c4d0" width="400"/>

## Train Custom Data
**相信这是大部分开发者最关心的事情，本章节参考如下仓库，本仓库现已集成**
* Converting darknet or yolov5 datasets to COCO format for YOLOX: [YOLO2COCO](https://github.com/RapidAI/YOLO2COCO) from [Daniel](https://github.com/znsoftm)

### 数据准备
**我们同样以YOLOv5格式的光栅数据集为例，可在[此处下载](https://aistudio.baidu.com/aistudio/datasetdetail/114547)**
进入仓库根目录，下载解压，数据集应该具有如下目录：
```
YOLOX-Paddle
|-- guangshan
|   |-- images
|      |-- train
|      |-- val
|   |-- labels
|      |-- train
|      |-- val
```
现在运行如下命令
```shell
bash prepare.sh
```
然后添加一个`classes.txt`，你应该得到如下目录，并在生成的`YOLOV5_COCO_format`得到COCO数据格式的数据集：
```
YOLOX-Paddle/YOLO2COCO/dataset
|-- YOLOV5
|   |-- guangshan
|   |   |-- images
|   |   |-- labels
|   |-- train.txt
|   |-- val.txt
|   |-- classes.txt
|-- YOLOV5_COCO_format
|   |-- train2017
|   |-- val2017
|   |-- annotations
```
可参考YOLOV5_COCO_format下的`README.md`

### 训练、验证、推理
配置custom训练文件`YOLOX-Paddle/exps/example/custom/nano.py`，修改`self.num_classes`为你的类别数，其余配置可根据喜好调参，使用如下命令启动训练
```
python tools/train.py -f ./exps/example/custom/nano.py -n yolox-nano -d 1 -b 8
```
使用如下命令启动验证
```
python tools/eval.py -f ./exps/example/custom/nano.py -n yolox-nano -c ./YOLOX_outputs_custom/nano/best_ckpt.pdparams -b 64 -d 1 --conf 0.001
```
使用如下命令启动推理
```
python tools/demo.py image -f ./exps/example/custom/nano.py -n yolox-nano -c ./YOLOX_outputs_custom/nano/best_ckpt.pdparams --path test.jpg --conf 0.25 --nms 0.45 --tsize 640 --save_result
```
**其余部分参考COCO数据集**，整个训练文件保存在`YOLOX_outputs_custom`文件夹

## **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| CSDN主页        | [Deep Hao的CSDN主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
| GitHub主页        | [Deep Hao的GitHub主页](https://github.com/GuoQuanhao) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
