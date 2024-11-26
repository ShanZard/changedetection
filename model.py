from pathlib import Path
from typing import Dict, Optional, Union
import torch
import torch.nn as nn
from models import *
import os.path as osp
import os
from utils import dataset
from torch.utils import data
from tqdm import tqdm
from utils.metrics import confusion
from torchvision.utils import save_image
from accelerate import Accelerator,DistributedDataParallelKwargs
# from accelerate.utils import tqdm
import torchvision.transforms as tfs
from datasets import load_dataset,load_from_disk
import evaluate
import models
import numpy as np
import time
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F
from accelerate.utils import set_seed
from torch.autograd import Variable
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt
import skimage.io as skio
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.reshape(size, -1)
        target_ = target.reshape(size, -1)

        return self.bceloss(pred_, target_)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))



        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        # print(logpt.shape,target.shape)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss

def dice_loss(logits, true, eps=1e-7):
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes).cuda()
        true_1_hot = true_1_hot[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

class DiceFocalLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(DiceFocalLoss, self).__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=None)
        self.dice = dice_loss
    def forward(self,predictions, target):
        focal = self.focal(predictions, target)
        dice = dice_loss(predictions, target)
        print(predictions)
        loss = dice + focal
        print(dice,focal)
        return loss

class Change_Detection_Framework(nn.Module, PyTorchModelHubMixin):
    def __init__(self,config):
        super().__init__()
        self.configs=config
        self._prepare_accelerator()
        
        self._get_device()
        self._get_CD_model_using_name()
        self.accelerator.print(self.configs)

    def training_CD(self):
        self.mode="train"
        self._get_optim()
        self._get_lr_scheduler()
        self._get_loss()
        self._get_CD_dataloader()
        
        best_f1=0
        save_path=osp.join(self.configs["test"]["save_path"],self.configs["dataset_name"],self.configs["model_name"])
        os.makedirs(save_path,exist_ok=True)  
        for epoch in range(self.configs["train"]["epochs"]):
            loss_list=[]
            self.CD_model.train()
            self.accelerator.print(f"Training Start, Current Epoch {epoch}")
            for i,batch in enumerate(tqdm(self.CD_dataloader_train, disable=not self.accelerator.is_local_main_process, miniters=20)):
                self.optimizer.zero_grad()
                pre_tensor, post_tensor, label_tensor, fname = batch["pre"], batch["post"], batch["gt"], batch["fname"]
                pre_tensor = pre_tensor.to(self.device)
                post_tensor = post_tensor.to(self.device)
                label_tensor = label_tensor.to(self.device)
                if label_tensor.shape[0]==1:
                    label_tensor.squeeze().unsqueeze()
                else:
                    label_tensor.squeeze()
                prediction = self.CD_model(pre_tensor, post_tensor)
                if self.configs["train"]["loss"]=="integrated": #MineNetCD/MaskCD
                    total_loss = self.CD_model(x1=pre_tensor,x2=post_tensor,labels=label_tensor.long()).loss
                else:
                    prediction = self.CD_model(pre_tensor, post_tensor)
                    if self.configs["train"]["loss"]=="BCE" or self.configs["train"]["loss"]=="Dice":
                        total_loss=self.loss(prediction["main_predictions"].squeeze(),label_tensor.float())
                    else:
                        total_loss=self.loss(prediction["main_predictions"].squeeze(),label_tensor.long())
                loss_list.append(total_loss.item()) #only append the loss value and ignore the grad to save memory
                self.accelerator.backward(total_loss)
                self.optimizer.step()
                if i%10 == 0 :
                    os.makedirs(os.path.join(save_path,'train', str(epoch)),exist_ok=True)     
                    probs = F.softmax(prediction['main_predictions'],dim=1)
                    probs=torch.argmax(probs,dim=1)
                    pred_show = probs[0,:,:].detach().cpu().numpy().copy()
                    pre_show = pre_tensor[0,:,:,:].detach().cpu().numpy().copy()
                    post_show = post_tensor[0,:,:,:].detach().cpu().numpy().copy()
                    label_tensor_show = label_tensor[0,:,:].detach().cpu().numpy().copy()
                    pred_show = np.expand_dims(pred_show, axis=-1)
                    pred_show = pred_show.repeat(repeats=3,axis=2)
                    pre_show = pre_show.transpose(1,2,0)
                    post_show = post_show.transpose(1,2,0)
                    label_tensor_show = np.expand_dims(label_tensor_show, axis=-1)
                    label_tensor_show = label_tensor_show.repeat(repeats=3,axis=2)
                    pred_show = np.array(pred_show*255,dtype=np.uint8)
                    label_tensor_show = np.array(label_tensor_show*255,dtype=np.uint8)
                    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                    for ax in axs.flat:
                        ax.axis('off')
                    axs[0, 0].imshow(pre_show)  
                    axs[0, 0].set_title('Pre-image')
                    axs[0, 1].imshow(post_show)
                    axs[0, 1].set_title('Post-image')
                    axs[1, 0].imshow(pred_show)
                    axs[1, 0].set_title('Predition')
                    axs[1, 1].imshow(label_tensor_show)
                    axs[1, 1].set_title('Label')
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path,'train', str(epoch),fname[0]))
                    plt.close()
                               
            loss_avg=sum(loss_list)/len(loss_list)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            print(f'Epoch {epoch} training completed, the average loss is {loss_avg}')


            if (epoch+1)%self.configs["train"]["save_intervals"]==0:
                self.current_epoch=epoch
                save_directory=osp.join(self.configs["test"]["save_path"],self.configs["dataset_name"],self.configs["model_name"],'checkpoints', str(epoch))
                save_best_directory=osp.join(self.configs["test"]["save_path"],self.configs["dataset_name"],self.configs["model_name"],'checkpoints', "BestF1")
                os.makedirs(save_directory,exist_ok=True)
                self.save_pretrained(save_directory=save_directory,config=self.configs,repo_id=f'{self.configs["dataset_name"]}_{self.configs["model_name"]}')
                if self.configs["eval"]:
                    f1=self.testing_CD(epoch)
                    if f1>best_f1:
                        self.save_pretrained(save_directory=save_best_directory,config=self.configs,repo_id=f'{self.configs["dataset_name"]}_{self.configs["model_name"]}',push_to_hub=self.configs["push_to_hub"])
                        best_f1=f1
                elif epoch+1==self.configs["train"]["epochs"]:
                    self.save_pretrained(save_directory=save_directory,config=self.configs,repo_id=f'{self.configs["dataset_name"]}_{self.configs["model_name"]}',push_to_hub=self.configs["push_to_hub"]) # push_to_hub last epoch

    # def evaluating_CD(self):
    #     print("evaluating:")
    #     self.CD_model.eval()
    #     self._get_metrics()
    #     # self.CD_dataloader_eval=Accelerator.prepare_data_loader(self.CD_dataloader_eval) # Disable Accelerator in eval to avoid multiple print (The "gather" method in Accelerator is very slow.)
    #     for _, batch in enumerate(self.CD_dataloader_eval):
    #         pre_tensor, post_tensor, label_tensor, fname = batch["pre"], batch["post"], batch["gt"], batch["fname"]
    #         pre_tensor = pre_tensor.to(self.device)
    #         post_tensor = post_tensor.to(self.device)
    #         label_tensor = label_tensor.to(self.device)
    #         probs=self.CD_model(pre_tensor, post_tensor)["main_predictions"]
    #         # prediction = torch.where(probs>0.5,1,0)
    #         if probs.shape[1]==1:
    #             prediction=torch.where(probs>0.5,1.0,0.0).squeeze()
    #         else:
    #             probs = F.softmax(probs,dim=1)
    #             prediction=probs.max(dim=1)[1]
    #         for i in range(prediction.shape[0]):
    #             # print(label_tensor[i].type(torch.int32).flatten().shape,prediction[i].max(dim=0)[1].flatten().shape)
    #             self.f1.add_batch(references=label_tensor[i].flatten(),predictions=prediction[i].flatten())
    #     f1=self.f1.compute()
        
    #     ts_f1=torch.FloatTensor([f1["f1"]]).cuda().unsqueeze(0)
    #     ts_f1_gathered=self.accelerator.gather(ts_f1)
    #     final_metric=torch.mean(ts_f1_gathered, dim=0)
    #     self.accelerator.print(f'evaluated f1 is {final_metric[0]}')
    #     return final_metric[0]

    def testing_CD(self,epoch):
        self.mode="test"
        print("testing:")
        self.CD_model.eval()
        self._get_metrics()
        self._get_CD_dataloader()
        save_path=osp.join(self.configs["test"]["save_path"],self.configs["dataset_name"],self.configs["model_name"])
        os.makedirs(save_path,exist_ok=True)
        TP,TN,FP,FN=0,0,0,0
        for _, data in enumerate(tqdm(self.CD_dataloader_test, disable=not self.accelerator.is_local_main_process, miniters=20)):
            pre_tensor, post_tensor, label_tensor, fname = data["pre"], data["post"], data["gt"], data["fname"]
            pre_tensor = pre_tensor.to(self.device)
            post_tensor = post_tensor.to(self.device)
            label_tensor = label_tensor.to(self.device)
            if "model" in self.configs and "type" in self.configs["model"]:
                if self.configs["model"]["type"]=="HG":
                    probs=self.CD_model(x1=pre_tensor, x2=post_tensor).logits
                if self.configs["model"]["type"]=="standard":
                    probs=self.CD_model(pre_tensor, post_tensor)["main_predictions"]
            else:
                probs=self.CD_model(pre_tensor, post_tensor)["main_predictions"]
            # if probs.shape[1]==1:
            #     prediction=torch.where(probs>0.5,1.0,0.0).squeeze()
            # else:
            probs = torch.nn.Softmax(dim=1)(probs)
            prediction=torch.argmax(probs,dim=1)
            tp,fp,tn,fn=confusion(prediction,label_tensor)
            assert tp+fp+tn+fn==prediction.shape.numel()
            TP+=tp
            TN+=tn
            FP+=fp
            FN+=fn

            ##############################save_image########################################
            os.makedirs(os.path.join(save_path,'test', str(epoch)),exist_ok=True)   
            os.makedirs(os.path.join(save_path,'test', str(epoch),'prediction'),exist_ok=True)
            os.makedirs(os.path.join(save_path,'test', str(epoch),'gt'),exist_ok=True)
            os.makedirs(os.path.join(save_path,'test', 'pre'),exist_ok=True)
            os.makedirs(os.path.join(save_path,'test', 'post'),exist_ok=True)

            for i in range(len(fname)):
                pred_show = prediction[i,:,:].detach().cpu().numpy().copy()
                pre_show = pre_tensor[i,:,:,:].detach().cpu().numpy().copy()
                post_show = post_tensor[i,:,:,:].detach().cpu().numpy().copy()
                label_tensor_show = label_tensor[i,:,:].detach().cpu().numpy().copy()
                pred_show = np.expand_dims(pred_show, axis=-1)
                pred_show = pred_show.repeat(repeats=3,axis=2)
                pre_show = pre_show.transpose(1,2,0)
                post_show = post_show.transpose(1,2,0)
                label_tensor_show = np.expand_dims(label_tensor_show, axis=-1)
                label_tensor_show = label_tensor_show.repeat(repeats=3,axis=2)
                pred_show = np.array(pred_show*255,dtype=np.uint8)
                label_tensor_show = np.array(label_tensor_show*255,dtype=np.uint8)
                pre_show = np.array(pre_show*255,dtype=np.uint8)
                post_show = np.array(post_show*255,dtype=np.uint8)
                skio.imsave(os.path.join(save_path,'test','pre',fname[i]),pre_show)
                skio.imsave(os.path.join(save_path,'test','post',fname[i]),post_show)
                skio.imsave(os.path.join(save_path,'test', str(epoch),'prediction',fname[i]),pred_show)
                skio.imsave(os.path.join(save_path,'test', str(epoch),'gt',fname[i]),label_tensor_show)

                # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                # for ax in axs.flat:
                #     ax.axis('off')
                # axs[0, 0].imshow(pre_show)  
                # axs[0, 0].set_title('Pre-image')
                # axs[0, 1].imshow(post_show)
                # axs[0, 1].set_title('Post-image')
                # axs[1, 0].imshow(pred_show)
                # axs[1, 0].set_title('Predition')
                # axs[1, 1].imshow(label_tensor_show)
                # axs[1, 1].set_title('Label')
                # plt.tight_layout()
                # plt.savefig(os.path.join(save_path,'train', str(epoch),fname[0]))
                # plt.close()            
        self.accelerator.print(TP, TN, FP, FN)
        OA=(TP+TN)/(TP+TN+FP+FN)
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        f1=2*TP/(2*TP+FP+FN)
        ciou=TP/(TP+FP+FN)
        ts_metrics_list=torch.FloatTensor([OA,f1,precision,recall,ciou]).cuda().unsqueeze(0)
        ts_eval_metric_gathered=self.accelerator.gather(ts_metrics_list)
        final_metric=torch.mean(ts_eval_metric_gathered, dim=0)
        self.accelerator.print(f'Accuracy={final_metric[0]:.04}, mF1={final_metric[1]:.04}, Precision={final_metric[2]:.04}, Recall={final_metric[3]:.04}, ciou={final_metric[4]:.04}')
        metrics_str = f'Epoch={epoch}, Accuracy={final_metric[0]:.04}, mF1={final_metric[1]:.04}, Precision={final_metric[2]:.04}, Recall={final_metric[3]:.04}, ciou={final_metric[4]:.04}\n'
        with open(os.path.join(save_path,'test','metrics.txt'), 'a') as file:
            file.write(metrics_str)
        
        return final_metric[4]

    def calculate_parameters(self):
        model=self.CD_model
        model.eval()
        input=(torch.randn(1,3,256,256).cuda(),torch.randn(1,3,256,256).cuda())
        flops = FlopCountAnalysis(model, input)
        params=sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.accelerator.print(f'pytorch params {params/1048576}M')
        self.accelerator.print(f'flops {flops.total()/1073741824}G')
        

    def confusion(prediction, truth):
        """ Returns the confusion matrix for the values in the `prediction` and `truth`
        tensors, i.e. the amount of positions where the values of `prediction`
        and `truth` are
        - 1 and 1 (True Positive)
        - 1 and 0 (False Positive)
        - 0 and 0 (True Negative)
        - 0 and 1 (False Negative)
        """

        confusion_vector = prediction / truth
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)

        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        return true_positives, false_positives, true_negatives, false_negatives

    def _get_metrics(self):
        self.metrics=evaluate.combine(["/private/evaluate/metrics/accuracy/accuracy.py", \
                                       "/private/evaluate/metrics/f1/f1.py",\
                                        "/private/evaluate/metrics/precision/precision.py", \
                                        "/private/evaluate/metrics/recall/recall.py"]) # accept flattened tensors
        self.mean_iou=evaluate.load("/private/evaluate/metrics/mean_iou/mean_iou.py") # accept listed and unflattened tensors (e.g., [tensor(shape=(h,w))])
        self.f1=evaluate.load("/private/evaluate/metrics/f1/f1.py")

    def _get_CD_model_using_name(self):
        model= models.find_model_using_name(self.configs)
        self.accelerator.print("model loaded")

        if "use_external_checkpoint" in self.configs["test"] and self.configs["test"]["use_external_checkpoint"]==True:
            self.accelerator.print("loading from external checkpoint")
            if self.configs["test"]["checkpoint_type"]=="HG":
                model=model.from_pretrained(self.configs["test"]["external_checkpoint"])
                self.accelerator.print(f'pretrained model loaded from {self.configs["test"]["external_checkpoint"]}')
            else:
                self.accelerator.print("this feature is untested!!!")
                model=model.load_state_dict(torch.load(self.configs["test"]["external_checkpoint"]))

        self.CD_model=model.to(self.device)
        self.CD_model=self.accelerator.prepare_model(self.CD_model)

    def _get_CD_dataloader(self):
        transform=self.__get_transform()
        self.accelerator.print(f'Building dataloader from {self.configs["dataset_path"]}')
        if self.configs["data_type"]=="local":
            self.__get_CD_dataloader_local()
        elif self.configs["data_type"]=="cloud" or self.configs["data_type"]=="HG":
            self.__get_CD_dataloader_HG()

    def __get_CD_dataloader_local(self):
        transform=self.__get_transform()
        full_dataset=load_from_disk(self.configs["dataset_path"])
        train_ds=full_dataset["train"]
        test_ds=full_dataset["test"]
        val_ds=full_dataset["val"]

        if self.mode=="train":
            CD_dataset_train=dataset.change_detection_dataset_HG(dataset=train_ds, transform=transform)
            CD_dataloader_train=data.DataLoader(dataset=CD_dataset_train,batch_size=self.configs["train"]["batch_size"],shuffle=True,num_workers=0,pin_memory=False,collate_fn=None)
            self.CD_dataloader_train=self.accelerator.prepare_data_loader(CD_dataloader_train)
            if self.configs["eval"]:
                CD_dataset_eval=dataset.change_detection_dataset_HG(dataset=test_ds, transform=transform)
                CD_dataloader_eval=data.DataLoader(dataset=CD_dataset_eval,batch_size=self.configs["eval"]["batch_size"],shuffle=False,num_workers=0,pin_memory=False,collate_fn=None)
                self.CD_dataloader_eval=self.accelerator.prepare(CD_dataloader_eval)

        elif self.mode=="test":
            CD_dataset_test=dataset.change_detection_dataset_HG(dataset=test_ds, transform=transform)
            CD_dataloader_test=data.DataLoader(dataset=CD_dataset_test,batch_size=self.configs["test"]["batch_size"],shuffle=False,num_workers=0,pin_memory=False,collate_fn=None)
            self.CD_dataloader_test=self.accelerator.prepare_data_loader(CD_dataloader_test)
    
    def __get_CD_dataloader_HG(self):
        self.accelerator.print("building dataloader")
        transform=self.__get_transform()
        full_dataset=load_dataset(self.configs["dataset_path"])

        # print(self.mode)

        if self.mode=="train":
            train_ds=full_dataset["train"]
            val_ds=full_dataset["test"]
            CD_dataset_train=dataset.change_detection_dataset_HG(dataset=train_ds, transform=transform)
            CD_dataloader_train=data.DataLoader(dataset=CD_dataset_train,batch_size=self.configs["train"]["batch_size"],shuffle=True,num_workers=0,pin_memory=False,collate_fn=None)
            self.CD_dataloader_train=self.accelerator.prepare_data_loader(CD_dataloader_train)
            if self.configs["eval"]:
                CD_dataset_eval=dataset.change_detection_dataset_HG(dataset=val_ds, transform=transform)
                CD_dataloader_eval=data.DataLoader(dataset=CD_dataset_eval,batch_size=self.configs["eval"]["batch_size"],shuffle=False,num_workers=0,pin_memory=False,collate_fn=None)
                self.CD_dataloader_eval=self.accelerator.prepare(CD_dataloader_eval)

        elif self.mode=="test":
            test_ds=full_dataset["test"]
            CD_dataset_test=dataset.change_detection_dataset_HG(dataset=test_ds, transform=transform)
            CD_dataloader_test=data.DataLoader(dataset=CD_dataset_test,batch_size=self.configs["test"]["batch_size"],shuffle=False,num_workers=0,pin_memory=False,collate_fn=None)
            self.CD_dataloader_test=self.accelerator.prepare_data_loader(CD_dataloader_test)


    def __get_transform(self):
        ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
        ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
        # ADE_MEAN = np.array([0,0,0])
        transform=[tfs.ToTensor()]

        
        if "transform" in self.configs:
            if "normalize" in self.configs["transform"]:
                transform.append(tfs.Normalize(mean=ADE_MEAN,std=ADE_STD))
        # self.accelerator.print(f'Data Transformation: {transform}')
        transform=tfs.Compose(transform)
        return transform

    def _get_optim(self):
        if self.configs["train"]["optim"]=="Adam":
            optimizer=torch.optim.Adam(self.CD_model.parameters(),lr=self.configs["train"]["lr"])
        elif self.configs["train"]["optim"]=="AdamW":
            optimizer=torch.optim.AdamW(self.CD_model.parameters(),lr=self.configs["train"]["lr"], weight_decay=self.configs["train"]["weight_decay"], amsgrad=False)
        self.optimizer=self.accelerator.prepare_optimizer(optimizer)

    def _get_lr_scheduler(self):
        if "lr_scheduler" not in self.configs["train"]:
            lr_scheduler=None
        elif self.configs["train"]["lr_scheduler"]=="cosine":
            lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,self.configs["train"]["epochs"])
            lr_scheduler=self.accelerator.prepare_scheduler(lr_scheduler)
        self.lr_scheduler=lr_scheduler

    def _get_loss(self):
        loss_list=[]
        if self.configs["train"]["loss"]=="BCE":
            loss=torch.nn.BCELoss()
        elif self.configs["train"]["loss"]=="CrossEntropy":
            loss=torch.nn.CrossEntropyLoss()
        elif self.configs["train"]["loss"]=="Dice":
            loss=BceDiceLoss()
        elif self.configs["train"]["loss"]=="DiceFocal":
            loss=DiceFocalLoss()
        elif self.configs["train"]["loss"]=="integrated":
            loss=None
        else:
            raise ValueError("specified loss function unsupported!!!")
        self.loss=loss
        
    def _get_device(self):
        if self.configs["device"]=="cuda":
            device="cuda"
        else:
            device="cpu"
        self.device=device

    def _prepare_accelerator(self):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator=Accelerator(kwargs_handlers=[ddp_kwargs])
        set_seed(8888)

class Distributed_CD_Model(nn.Module):
    def __init__(self,configs) -> None:
        super().__init__()
        self.configs=configs
        pass
    def _get_encoder(self):
        pass
    def _get_decoder(self):
        pass
    def forward(self,*args,**kargs):
        encoded_features=self.encoder(*args,**kargs)
        decoded_features=self.decoder(encoded_features)
        if self.configs["Model"]["Sigmoid"]=="True":
             decoded_features=torch.nn.Sigmoid(decoded_features)
        return decoded_features

