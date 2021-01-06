# mask_rcnn.res50.c4.coco.multiscale.2x.syncbn.extra_norm  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.597
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.435
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.230
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.452
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.552
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.600
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.714
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 40.101 | 59.714 | 43.539 | 22.974 | 45.199 | 55.214 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 55.086 | bicycle      | 31.861 | car            | 42.149 |  
| motorcycle    | 42.645 | airplane     | 63.896 | bus            | 65.363 |  
| train         | 61.097 | truck        | 34.056 | boat           | 27.044 |  
| traffic light | 25.498 | fire hydrant | 68.326 | stop sign      | 64.346 |  
| parking meter | 46.667 | bench        | 24.461 | bird           | 34.429 |  
| cat           | 62.744 | dog          | 58.212 | horse          | 58.404 |  
| sheep         | 45.829 | cow          | 51.549 | elephant       | 65.058 |  
| bear          | 68.431 | zebra        | 67.633 | giraffe        | 67.564 |  
| backpack      | 15.081 | umbrella     | 39.992 | handbag        | 14.450 |  
| tie           | 31.390 | suitcase     | 36.723 | frisbee        | 61.551 |  
| skis          | 24.469 | snowboard    | 34.636 | sports ball    | 41.538 |  
| kite          | 38.479 | baseball bat | 25.133 | baseball glove | 35.963 |  
| skateboard    | 52.636 | surfboard    | 37.195 | tennis racket  | 46.828 |  
| bottle        | 37.423 | wine glass   | 33.913 | cup            | 40.964 |  
| fork          | 34.441 | knife        | 14.575 | spoon          | 15.540 |  
| bowl          | 40.900 | banana       | 24.020 | apple          | 18.839 |  
| sandwich      | 33.077 | orange       | 29.684 | broccoli       | 16.577 |  
| carrot        | 16.101 | hot dog      | 34.212 | pizza          | 51.124 |  
| donut         | 43.455 | cake         | 33.643 | chair          | 27.525 |  
| couch         | 41.603 | potted plant | 27.135 | bed            | 37.526 |  
| dining table  | 27.277 | toilet       | 57.345 | tv             | 55.553 |  
| laptop        | 61.295 | mouse        | 57.564 | remote         | 26.249 |  
| keyboard      | 52.829 | cell phone   | 31.825 | microwave      | 55.978 |  
| oven          | 32.778 | toaster      | 39.930 | sink           | 36.562 |  
| refrigerator  | 53.819 | book         | 12.765 | clock          | 47.181 |  
| vase          | 36.357 | scissors     | 29.258 | teddy bear     | 47.083 |  
| hair drier    | 9.599  | toothbrush   | 16.123 |                |        |
## Evaluation results for segm:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.564
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.155
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.384
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.532
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.459
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.286
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.650
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 34.690 | 56.419 | 37.116 | 15.546 | 38.381 | 53.233 |
### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 44.738 | bicycle      | 18.734 | car            | 37.910 |  
| motorcycle    | 33.345 | airplane     | 41.919 | bus            | 62.332 |  
| train         | 59.833 | truck        | 33.332 | boat           | 21.947 |  
| traffic light | 24.197 | fire hydrant | 63.113 | stop sign      | 61.631 |  
| parking meter | 46.120 | bench        | 16.097 | bird           | 25.922 |  
| cat           | 62.834 | dog          | 53.631 | horse          | 38.133 |  
| sheep         | 38.170 | cow          | 41.430 | elephant       | 55.926 |  
| bear          | 66.666 | zebra        | 53.879 | giraffe        | 46.262 |  
| backpack      | 13.837 | umbrella     | 42.970 | handbag        | 11.814 |  
| tie           | 26.636 | suitcase     | 37.664 | frisbee        | 57.051 |  
| skis          | 2.014  | snowboard    | 19.789 | sports ball    | 40.479 |  
| kite          | 24.170 | baseball bat | 19.914 | baseball glove | 37.418 |  
| skateboard    | 30.660 | surfboard    | 28.882 | tennis racket  | 50.527 |  
| bottle        | 34.976 | wine glass   | 28.843 | cup            | 40.063 |  
| fork          | 14.881 | knife        | 9.135  | spoon          | 9.649  |  
| bowl          | 36.737 | banana       | 18.893 | apple          | 17.342 |  
| sandwich      | 34.168 | orange       | 28.245 | broccoli       | 15.705 |  
| carrot        | 12.495 | hot dog      | 28.866 | pizza          | 48.620 |  
| donut         | 42.950 | cake         | 34.399 | chair          | 17.701 |  
| couch         | 33.372 | potted plant | 22.236 | bed            | 28.008 |  
| dining table  | 15.504 | toilet       | 54.403 | tv             | 57.888 |  
| laptop        | 59.073 | mouse        | 56.345 | remote         | 20.818 |  
| keyboard      | 48.996 | cell phone   | 29.146 | microwave      | 56.370 |  
| oven          | 30.595 | toaster      | 44.239 | sink           | 32.090 |  
| refrigerator  | 54.878 | book         | 6.815  | clock          | 48.216 |  
| vase          | 34.699 | scissors     | 19.906 | teddy bear     | 44.475 |  
| hair drier    | 2.653  | toothbrush   | 8.918  |                |        |
