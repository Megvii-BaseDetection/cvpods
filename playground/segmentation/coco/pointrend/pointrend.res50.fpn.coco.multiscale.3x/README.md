# pointrend.res50.fpn.coco.multiscale.3x  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.616
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.252
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.441
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.524
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.365
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.685
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 41.053 | 61.618 | 44.846 | 25.247 | 44.062 | 53.463 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 55.670 | bicycle      | 30.488 | car            | 44.867 |  
| motorcycle    | 42.748 | airplane     | 63.041 | bus            | 63.971 |  
| train         | 62.769 | truck        | 35.562 | boat           | 27.572 |  
| traffic light | 28.083 | fire hydrant | 67.198 | stop sign      | 66.696 |  
| parking meter | 44.381 | bench        | 24.178 | bird           | 37.113 |  
| cat           | 68.038 | dog          | 59.962 | horse          | 58.025 |  
| sheep         | 49.707 | cow          | 54.503 | elephant       | 62.196 |  
| bear          | 66.140 | zebra        | 66.463 | giraffe        | 67.079 |  
| backpack      | 16.005 | umbrella     | 38.840 | handbag        | 14.456 |  
| tie           | 32.471 | suitcase     | 37.223 | frisbee        | 63.124 |  
| skis          | 23.174 | snowboard    | 37.901 | sports ball    | 47.165 |  
| kite          | 41.823 | baseball bat | 25.950 | baseball glove | 35.224 |  
| skateboard    | 51.502 | surfboard    | 38.153 | tennis racket  | 46.017 |  
| bottle        | 39.371 | wine glass   | 35.589 | cup            | 42.242 |  
| fork          | 34.708 | knife        | 18.790 | spoon          | 18.031 |  
| bowl          | 42.618 | banana       | 23.155 | apple          | 19.646 |  
| sandwich      | 33.430 | orange       | 30.477 | broccoli       | 23.159 |  
| carrot        | 21.986 | hot dog      | 30.933 | pizza          | 50.968 |  
| donut         | 41.863 | cake         | 33.688 | chair          | 27.278 |  
| couch         | 41.851 | potted plant | 27.328 | bed            | 39.850 |  
| dining table  | 27.433 | toilet       | 57.817 | tv             | 55.889 |  
| laptop        | 59.181 | mouse        | 61.768 | remote         | 29.810 |  
| keyboard      | 51.109 | cell phone   | 35.648 | microwave      | 55.423 |  
| oven          | 32.292 | toaster      | 49.132 | sink           | 36.116 |  
| refrigerator  | 54.639 | book         | 14.982 | clock          | 49.994 |  
| vase          | 38.224 | scissors     | 28.236 | teddy bear     | 44.825 |  
| hair drier    | 8.020  | toothbrush   | 21.277 |                |        |


## Evaluation results for segm:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.592
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.407
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.668
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 38.238 | 59.207 | 41.116 | 18.885 | 40.709 | 54.989 |

### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 48.978 | bicycle      | 18.633 | car            | 42.353 |  
| motorcycle    | 35.036 | airplane     | 53.048 | bus            | 64.139 |  
| train         | 64.246 | truck        | 36.153 | boat           | 24.982 |  
| traffic light | 26.985 | fire hydrant | 64.005 | stop sign      | 66.730 |  
| parking meter | 45.198 | bench        | 17.929 | bird           | 31.597 |  
| cat           | 70.131 | dog          | 59.374 | horse          | 45.092 |  
| sheep         | 44.375 | cow          | 48.293 | elephant       | 59.432 |  
| bear          | 65.899 | zebra        | 59.173 | giraffe        | 55.656 |  
| backpack      | 16.773 | umbrella     | 45.866 | handbag        | 14.888 |  
| tie           | 32.113 | suitcase     | 39.655 | frisbee        | 62.706 |  
| skis          | 3.849  | snowboard    | 24.950 | sports ball    | 47.063 |  
| kite          | 31.468 | baseball bat | 25.484 | baseball glove | 38.046 |  
| skateboard    | 34.094 | surfboard    | 33.589 | tennis racket  | 54.560 |  
| bottle        | 38.493 | wine glass   | 32.857 | cup            | 43.105 |  
| fork          | 18.213 | knife        | 13.553 | spoon          | 12.694 |  
| bowl          | 39.417 | banana       | 19.660 | apple          | 20.105 |  
| sandwich      | 37.189 | orange       | 30.827 | broccoli       | 22.762 |  
| carrot        | 19.458 | hot dog      | 24.478 | pizza          | 51.287 |  
| donut         | 43.277 | cake         | 35.174 | chair          | 19.380 |  
| couch         | 36.209 | potted plant | 23.822 | bed            | 31.803 |  
| dining table  | 15.576 | toilet       | 58.878 | tv             | 59.422 |  
| laptop        | 60.550 | mouse        | 62.731 | remote         | 29.369 |  
| keyboard      | 51.754 | cell phone   | 35.052 | microwave      | 57.843 |  
| oven          | 30.707 | toaster      | 53.622 | sink           | 35.466 |  
| refrigerator  | 57.286 | book         | 10.737 | clock          | 50.602 |  
| vase          | 37.309 | scissors     | 20.961 | teddy bear     | 44.727 |  
| hair drier    | 6.296  | toothbrush   | 13.849 |                |        |
