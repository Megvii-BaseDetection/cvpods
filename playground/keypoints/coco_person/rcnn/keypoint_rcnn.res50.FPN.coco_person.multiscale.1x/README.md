# keypoint_rcnn.res50.FPN.coco_person.multiscale.1x  

seed: 14924736

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.537
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.823
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.584
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.616
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.707
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.188
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.548
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.630
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.691
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.778
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 53.684 | 82.315 | 58.375 | 36.169 | 61.614 | 70.660 |

### Per-category bbox AP:  

| category   | AP     |  
|:-----------|:-------|  
| person     | 53.684 |


## Evaluation results for keypoints:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.642
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.862
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.695
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.598
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.724
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.712
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.912
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.762
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.656
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.790
```  
|   AP   |  AP50  |  AP75  |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|  
| 64.162 | 86.248 | 69.455 | 59.772 | 72.356 |

### Per-category keypoints AP:  

| category   | AP     |  
|:-----------|:-------|  
| person     | 64.162 |
