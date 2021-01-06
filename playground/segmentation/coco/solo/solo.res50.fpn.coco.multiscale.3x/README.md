# solo.res50.fpn.coco.multiscale.3x  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.564
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.174
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.568
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.714
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 35.592 | 56.363 | 37.516 | 17.376 | 39.114 | 51.332 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 45.647 | bicycle      | 24.018 | car            | 34.419 |  
| motorcycle    | 37.495 | airplane     | 57.508 | bus            | 63.180 |  
| train         | 66.664 | truck        | 33.286 | boat           | 22.090 |  
| traffic light | 21.946 | fire hydrant | 57.797 | stop sign      | 61.016 |  
| parking meter | 41.627 | bench        | 19.573 | bird           | 27.051 |  
| cat           | 63.793 | dog          | 55.073 | horse          | 51.497 |  
| sheep         | 46.838 | cow          | 46.259 | elephant       | 58.167 |  
| bear          | 67.767 | zebra        | 61.589 | giraffe        | 60.924 |  
| backpack      | 10.905 | umbrella     | 26.716 | handbag        | 7.586  |  
| tie           | 28.126 | suitcase     | 33.997 | frisbee        | 54.439 |  
| skis          | 19.774 | snowboard    | 26.586 | sports ball    | 31.085 |  
| kite          | 25.278 | baseball bat | 16.037 | baseball glove | 29.948 |  
| skateboard    | 44.477 | surfboard    | 28.074 | tennis racket  | 34.084 |  
| bottle        | 27.948 | wine glass   | 26.788 | cup            | 34.451 |  
| fork          | 24.843 | knife        | 13.854 | spoon          | 11.254 |  
| bowl          | 33.379 | banana       | 20.332 | apple          | 15.415 |  
| sandwich      | 32.397 | orange       | 23.528 | broccoli       | 20.485 |  
| carrot        | 20.919 | hot dog      | 30.500 | pizza          | 44.421 |  
| donut         | 41.332 | cake         | 31.339 | chair          | 21.084 |  
| couch         | 41.755 | potted plant | 23.015 | bed            | 43.483 |  
| dining table  | 24.459 | toilet       | 58.419 | tv             | 53.057 |  
| laptop        | 53.366 | mouse        | 51.595 | remote         | 22.091 |  
| keyboard      | 48.592 | cell phone   | 29.690 | microwave      | 53.233 |  
| oven          | 32.061 | toaster      | 29.345 | sink           | 33.130 |  
| refrigerator  | 56.656 | book         | 8.165  | clock          | 45.864 |  
| vase          | 33.561 | scissors     | 20.898 | teddy bear     | 44.147 |  
| hair drier    | 10.308 | toothbrush   | 13.832 |                |        |


## Evaluation results for segm:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.566
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.160
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.381
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.459
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.280
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.657
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 35.157 | 56.575 | 37.159 | 15.964 | 38.130 | 52.230 |

### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 42.111 | bicycle      | 16.993 | car            | 36.043 |  
| motorcycle    | 32.387 | airplane     | 50.067 | bus            | 62.333 |  
| train         | 63.210 | truck        | 35.024 | boat           | 21.088 |  
| traffic light | 24.207 | fire hydrant | 59.030 | stop sign      | 61.406 |  
| parking meter | 44.580 | bench        | 15.589 | bird           | 26.909 |  
| cat           | 65.701 | dog          | 54.411 | horse          | 40.257 |  
| sheep         | 43.897 | cow          | 45.450 | elephant       | 58.043 |  
| bear          | 69.553 | zebra        | 57.290 | giraffe        | 49.605 |  
| backpack      | 12.697 | umbrella     | 42.232 | handbag        | 12.224 |  
| tie           | 28.834 | suitcase     | 36.585 | frisbee        | 59.673 |  
| skis          | 3.257  | snowboard    | 20.697 | sports ball    | 38.981 |  
| kite          | 27.041 | baseball bat | 21.279 | baseball glove | 36.983 |  
| skateboard    | 32.163 | surfboard    | 25.914 | tennis racket  | 50.038 |  
| bottle        | 29.411 | wine glass   | 28.432 | cup            | 36.855 |  
| fork          | 14.154 | knife        | 11.075 | spoon          | 9.006  |  
| bowl          | 32.233 | banana       | 17.052 | apple          | 17.098 |  
| sandwich      | 35.084 | orange       | 24.386 | broccoli       | 21.372 |  
| carrot        | 19.607 | hot dog      | 25.112 | pizza          | 44.828 |  
| donut         | 43.005 | cake         | 32.866 | chair          | 17.310 |  
| couch         | 36.917 | potted plant | 21.562 | bed            | 32.899 |  
| dining table  | 14.393 | toilet       | 60.795 | tv             | 55.264 |  
| laptop        | 57.250 | mouse        | 56.022 | remote         | 25.543 |  
| keyboard      | 49.452 | cell phone   | 32.121 | microwave      | 55.734 |  
| oven          | 32.203 | toaster      | 35.955 | sink           | 34.569 |  
| refrigerator  | 54.487 | book         | 5.241  | clock          | 48.774 |  
| vase          | 34.615 | scissors     | 16.439 | teddy bear     | 45.210 |  
| hair drier    | 10.656 | toothbrush   | 13.795 |                |        |
