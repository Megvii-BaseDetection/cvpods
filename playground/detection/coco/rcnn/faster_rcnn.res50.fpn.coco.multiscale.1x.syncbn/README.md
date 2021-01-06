# faster_rcnn.res50.fpn.coco.multiscale.2x.syncbn  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.606
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.430
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.326
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.512
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.537
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.674
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 39.901 | 60.559 | 43.417 | 23.748 | 42.996 | 51.868 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 54.303 | bicycle      | 29.595 | car            | 44.519 |  
| motorcycle    | 43.286 | airplane     | 62.084 | bus            | 61.796 |  
| train         | 60.153 | truck        | 36.737 | boat           | 28.027 |  
| traffic light | 27.839 | fire hydrant | 65.501 | stop sign      | 68.143 |  
| parking meter | 44.333 | bench        | 22.627 | bird           | 35.647 |  
| cat           | 65.003 | dog          | 58.060 | horse          | 55.182 |  
| sheep         | 49.021 | cow          | 53.302 | elephant       | 58.557 |  
| bear          | 64.229 | zebra        | 65.472 | giraffe        | 66.358 |  
| backpack      | 15.222 | umbrella     | 38.092 | handbag        | 14.043 |  
| tie           | 32.573 | suitcase     | 36.905 | frisbee        | 62.349 |  
| skis          | 22.162 | snowboard    | 36.960 | sports ball    | 47.259 |  
| kite          | 40.800 | baseball bat | 26.658 | baseball glove | 35.185 |  
| skateboard    | 51.385 | surfboard    | 35.516 | tennis racket  | 44.958 |  
| bottle        | 38.877 | wine glass   | 34.866 | cup            | 41.227 |  
| fork          | 31.592 | knife        | 17.556 | spoon          | 15.656 |  
| bowl          | 40.965 | banana       | 21.732 | apple          | 18.894 |  
| sandwich      | 32.350 | orange       | 29.474 | broccoli       | 20.667 |  
| carrot        | 20.484 | hot dog      | 29.047 | pizza          | 50.527 |  
| donut         | 43.252 | cake         | 34.589 | chair          | 25.839 |  
| couch         | 39.526 | potted plant | 25.046 | bed            | 39.684 |  
| dining table  | 26.944 | toilet       | 55.835 | tv             | 54.206 |  
| laptop        | 57.893 | mouse        | 61.888 | remote         | 29.530 |  
| keyboard      | 49.344 | cell phone   | 34.492 | microwave      | 54.933 |  
| oven          | 31.445 | toaster      | 37.627 | sink           | 34.997 |  
| refrigerator  | 54.140 | book         | 15.326 | clock          | 49.330 |  
| vase          | 37.474 | scissors     | 23.201 | teddy bear     | 45.710 |  
| hair drier    | 6.139  | toothbrush   | 19.965 |                |        |
