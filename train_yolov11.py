from ultralytics import YOLO
import os
from utils.yolo.attention import add_attention
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

root = os.getcwd()
## 配置文件路径
name_yaml = os.path.join(root, "dataset2548.yaml")
## 约束训练路径、剪枝模型文件
name_prune_before =r"C:\Desktop\graduate_design\ultralytics-main\runs_ghost_new\stage3_full\weights\best.pt"
name_prune_after = os.path.join(root,r"exp_2548\weights\pruneandghost.pt")

## 微调路径
path_fineturn = os.path.join(root,r"exp_2548\weights\fintune_pruneandghost")

def step1_pruning():
    # from utils.yolo.seg_pruning import do_pruning  use for seg
    from utils.yolo.det_pruning import do_pruning  # use for det
    # do_pruning(name_prune_before, name_prune_after)
    do_pruning(name_prune_before, name_prune_after)


def step2_finetune():
    model = YOLO(name_prune_after)  # load a pretrained model (recommended for training)
    for param in model.parameters():
        param.requires_grad = True
    model.train(data=name_yaml, device=0, imgsz=640, epochs=200, batch=16, workers=0, name=path_fineturn)  # train the model



if __name__ == '__main__':
   step1_pruning()
   step2_finetune()
