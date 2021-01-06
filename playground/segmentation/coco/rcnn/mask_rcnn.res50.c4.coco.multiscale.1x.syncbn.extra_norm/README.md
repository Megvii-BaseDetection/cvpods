# mask_rcnn.res50.c4.coco.multiscale.1x.syncbn.extra_norm  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.379
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.579
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.414
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.201
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.507
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.324
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 37.920 | 57.934 | 41.370 | 20.080 | 43.295 | 51.645 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 53.553 | bicycle      | 29.752 | car            | 39.807 |  
| motorcycle    | 40.620 | airplane     | 62.866 | bus            | 62.820 |  
| train         | 57.185 | truck        | 31.610 | boat           | 24.640 |  
| traffic light | 24.093 | fire hydrant | 62.894 | stop sign      | 63.544 |  
| parking meter | 46.266 | bench        | 21.414 | bird           | 31.495 |  
| cat           | 62.489 | dog          | 58.571 | horse          | 55.623 |  
| sheep         | 47.827 | cow          | 52.105 | elephant       | 60.510 |  
| bear          | 66.311 | zebra        | 65.312 | giraffe        | 67.018 |  
| backpack      | 13.841 | umbrella     | 36.519 | handbag        | 12.976 |  
| tie           | 26.770 | suitcase     | 32.069 | frisbee        | 57.969 |  
| skis          | 19.263 | snowboard    | 33.075 | sports ball    | 40.296 |  
| kite          | 34.263 | baseball bat | 22.676 | baseball glove | 32.398 |  
| skateboard    | 49.391 | surfboard    | 33.833 | tennis racket  | 44.542 |  
| bottle        | 34.503 | wine glass   | 32.590 | cup            | 38.794 |  
| fork          | 29.793 | knife        | 10.715 | spoon          | 14.163 |  
| bowl          | 39.413 | banana       | 23.336 | apple          | 18.467 |  
| sandwich      | 31.598 | orange       | 28.731 | broccoli       | 18.725 |  
| carrot        | 19.171 | hot dog      | 31.222 | pizza          | 50.621 |  
| donut         | 42.888 | cake         | 29.877 | chair          | 25.125 |  
| couch         | 38.991 | potted plant | 24.596 | bed            | 35.650 |  
| dining table  | 24.880 | toilet       | 56.279 | tv             | 53.557 |  
| laptop        | 59.106 | mouse        | 55.500 | remote         | 22.925 |  
| keyboard      | 50.627 | cell phone   | 30.504 | microwave      | 56.409 |  
| oven          | 30.375 | toaster      | 38.830 | sink           | 33.877 |  
| refrigerator  | 49.459 | book         | 11.819 | clock          | 48.110 |  
| vase          | 33.134 | scissors     | 25.519 | teddy bear     | 42.095 |  
| hair drier    | 0.870  | toothbrush   | 12.579 |                |        |
## Evaluation results for segm:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.545
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.351
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.509
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.291
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.468
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.265
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.640
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 33.148 | 54.503 | 35.056 | 13.587 | 36.852 | 50.891 |
### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 43.120 | bicycle      | 16.630 | car            | 36.041 |  
| motorcycle    | 30.979 | airplane     | 40.716 | bus            | 61.502 |  
| train         | 57.159 | truck        | 31.037 | boat           | 20.433 |  
| traffic light | 22.701 | fire hydrant | 59.344 | stop sign      | 62.619 |  
| parking meter | 46.735 | bench        | 14.032 | bird           | 23.311 |  
| cat           | 62.780 | dog          | 54.341 | horse          | 35.905 |  
| sheep         | 39.519 | cow          | 41.467 | elephant       | 54.029 |  
| bear          | 65.759 | zebra        | 52.038 | giraffe        | 46.938 |  
| backpack      | 12.418 | umbrella     | 41.023 | handbag        | 10.535 |  
| tie           | 22.796 | suitcase     | 33.892 | frisbee        | 54.824 |  
| skis          | 0.890  | snowboard    | 18.675 | sports ball    | 38.886 |  
| kite          | 21.976 | baseball bat | 18.179 | baseball glove | 33.277 |  
| skateboard    | 26.564 | surfboard    | 25.890 | tennis racket  | 50.227 |  
| bottle        | 32.305 | wine glass   | 27.829 | cup            | 38.223 |  
| fork          | 11.630 | knife        | 7.263  | spoon          | 8.214  |  
| bowl          | 35.373 | banana       | 18.481 | apple          | 16.641 |  
| sandwich      | 32.698 | orange       | 27.518 | broccoli       | 17.031 |  
| carrot        | 15.061 | hot dog      | 25.812 | pizza          | 48.580 |  
| donut         | 42.228 | cake         | 30.889 | chair          | 15.185 |  
| couch         | 32.461 | potted plant | 20.533 | bed            | 27.086 |  
| dining table  | 14.402 | toilet       | 54.498 | tv             | 55.164 |  
| laptop        | 56.626 | mouse        | 53.907 | remote         | 19.192 |  
| keyboard      | 48.901 | cell phone   | 28.119 | microwave      | 55.981 |  
| oven          | 27.249 | toaster      | 42.612 | sink           | 31.038 |  
| refrigerator  | 50.295 | book         | 5.690  | clock          | 48.071 |  
| vase          | 30.759 | scissors     | 16.287 | teddy bear     | 40.884 |  
| hair drier    | 1.204  | toothbrush   | 10.701 |                |        |
