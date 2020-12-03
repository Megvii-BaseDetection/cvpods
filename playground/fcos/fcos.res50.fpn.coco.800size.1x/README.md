# fcos.res50.fpn.coco.800size.1x

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.575
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.226
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.427
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.534
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.720
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 38.719 | 57.490 | 53.539 | 46.949 | 41.692 | 35.188 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |
|:--------------|:-------|:-------------|:-------|:---------------|:-------|
| person        | 53.282 | bicycle      | 29.202 | car            | 42.108 |
| motorcycle    | 38.985 | airplane     | 63.561 | bus            | 63.563 |
| train         | 57.491 | truck        | 31.925 | boat           | 21.811 |
| traffic light | 25.712 | fire hydrant | 64.749 | stop sign      | 64.309 |
| parking meter | 42.604 | bench        | 20.516 | bird           | 34.071 |
| cat           | 64.635 | dog          | 61.370 | horse          | 52.967 |
| sheep         | 51.680 | cow          | 56.638 | elephant       | 62.866 |
| bear          | 71.860 | zebra        | 66.408 | giraffe        | 65.609 |
| backpack      | 14.180 | umbrella     | 37.956 | handbag        | 13.694 |
| tie           | 27.788 | suitcase     | 34.438 | frisbee        | 64.418 |
| skis          | 18.604 | snowboard    | 29.170 | sports ball    | 45.519 |
| kite          | 40.818 | baseball bat | 25.096 | baseball glove | 34.597 |
| skateboard    | 47.097 | surfboard    | 30.247 | tennis racket  | 45.019 |
| bottle        | 35.221 | wine glass   | 34.911 | cup            | 41.024 |
| fork          | 27.317 | knife        | 14.199 | spoon          | 13.405 |
| bowl          | 39.203 | banana       | 23.568 | apple          | 18.976 |
| sandwich      | 31.181 | orange       | 31.731 | broccoli       | 22.755 |
| carrot        | 19.831 | hot dog      | 27.377 | pizza          | 49.126 |
| donut         | 43.932 | cake         | 35.290 | chair          | 26.128 |
| couch         | 41.758 | potted plant | 26.208 | bed            | 39.194 |
| dining table  | 25.756 | toilet       | 57.755 | tv             | 51.880 |
| laptop        | 55.564 | mouse        | 59.272 | remote         | 26.730 |
| keyboard      | 44.607 | cell phone   | 33.582 | microwave      | 58.325 |
| oven          | 29.844 | toaster      | 26.954 | sink           | 33.510 |
| refrigerator  | 51.442 | book         | 12.788 | clock          | 48.099 |
| vase          | 34.989 | scissors     | 19.893 | teddy bear     | 43.942 |
| hair drier    | 7.573  | toothbrush   | 16.101 |                |        |
