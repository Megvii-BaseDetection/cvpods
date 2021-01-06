# yolov3.darknet53.coco.multiscale.syncbn  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.596
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.226
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.403
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.295
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.587
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 37.492 | 59.628 | 40.436 | 22.554 | 40.278 | 47.713 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 51.021 | bicycle      | 30.212 | car            | 40.934 |  
| motorcycle    | 40.661 | airplane     | 61.233 | bus            | 60.516 |  
| train         | 58.652 | truck        | 31.748 | boat           | 26.009 |  
| traffic light | 24.076 | fire hydrant | 63.751 | stop sign      | 63.089 |  
| parking meter | 37.279 | bench        | 22.120 | bird           | 34.721 |  
| cat           | 60.358 | dog          | 54.915 | horse          | 54.367 |  
| sheep         | 50.705 | cow          | 52.568 | elephant       | 58.274 |  
| bear          | 64.584 | zebra        | 62.893 | giraffe        | 62.907 |  
| backpack      | 13.028 | umbrella     | 36.377 | handbag        | 14.772 |  
| tie           | 28.957 | suitcase     | 33.462 | frisbee        | 59.455 |  
| skis          | 21.017 | snowboard    | 28.728 | sports ball    | 43.698 |  
| kite          | 39.266 | baseball bat | 26.104 | baseball glove | 34.449 |  
| skateboard    | 49.466 | surfboard    | 35.153 | tennis racket  | 45.231 |  
| bottle        | 35.035 | wine glass   | 31.671 | cup            | 37.655 |  
| fork          | 29.301 | knife        | 15.229 | spoon          | 13.963 |  
| bowl          | 36.151 | banana       | 20.962 | apple          | 18.513 |  
| sandwich      | 31.043 | orange       | 27.875 | broccoli       | 20.005 |  
| carrot        | 18.086 | hot dog      | 27.307 | pizza          | 44.308 |  
| donut         | 45.364 | cake         | 32.469 | chair          | 26.617 |  
| couch         | 37.146 | potted plant | 22.185 | bed            | 34.937 |  
| dining table  | 23.472 | toilet       | 53.205 | tv             | 51.972 |  
| laptop        | 53.083 | mouse        | 60.257 | remote         | 28.739 |  
| keyboard      | 45.099 | cell phone   | 31.312 | microwave      | 48.182 |  
| oven          | 28.263 | toaster      | 33.201 | sink           | 34.516 |  
| refrigerator  | 48.118 | book         | 9.275  | clock          | 49.643 |  
| vase          | 31.955 | scissors     | 24.074 | teddy bear     | 41.402 |  
| hair drier    | 0.000  | toothbrush   | 21.032 |                |        |
