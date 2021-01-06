# mask_rcnn.res50.c4.coco.multiscale.1x  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.566
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.518
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.576
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.680
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 36.828 | 56.639 | 39.945 | 20.185 | 41.721 | 50.410 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 52.172 | bicycle      | 28.793 | car            | 39.129 |  
| motorcycle    | 38.061 | airplane     | 61.152 | bus            | 60.915 |  
| train         | 57.490 | truck        | 29.044 | boat           | 23.486 |  
| traffic light | 23.703 | fire hydrant | 62.720 | stop sign      | 61.773 |  
| parking meter | 43.324 | bench        | 21.070 | bird           | 29.924 |  
| cat           | 62.987 | dog          | 54.415 | horse          | 55.056 |  
| sheep         | 44.592 | cow          | 49.807 | elephant       | 58.218 |  
| bear          | 64.235 | zebra        | 65.901 | giraffe        | 65.095 |  
| backpack      | 13.487 | umbrella     | 34.588 | handbag        | 10.377 |  
| tie           | 26.851 | suitcase     | 28.349 | frisbee        | 58.078 |  
| skis          | 17.145 | snowboard    | 33.238 | sports ball    | 40.627 |  
| kite          | 35.138 | baseball bat | 22.867 | baseball glove | 32.638 |  
| skateboard    | 47.694 | surfboard    | 32.559 | tennis racket  | 43.161 |  
| bottle        | 33.903 | wine glass   | 30.228 | cup            | 37.710 |  
| fork          | 27.442 | knife        | 9.999  | spoon          | 12.259 |  
| bowl          | 37.974 | banana       | 22.342 | apple          | 16.942 |  
| sandwich      | 28.936 | orange       | 27.458 | broccoli       | 21.321 |  
| carrot        | 19.712 | hot dog      | 28.933 | pizza          | 49.928 |  
| donut         | 40.533 | cake         | 30.780 | chair          | 24.339 |  
| couch         | 35.879 | potted plant | 22.951 | bed            | 34.188 |  
| dining table  | 23.630 | toilet       | 55.520 | tv             | 52.218 |  
| laptop        | 57.769 | mouse        | 53.397 | remote         | 22.440 |  
| keyboard      | 49.619 | cell phone   | 29.300 | microwave      | 53.906 |  
| oven          | 29.991 | toaster      | 42.308 | sink           | 30.184 |  
| refrigerator  | 48.720 | book         | 11.303 | clock          | 47.011 |  
| vase          | 32.140 | scissors     | 25.327 | teddy bear     | 42.568 |  
| hair drier    | 1.377  | toothbrush   | 13.928 |                |        |
## Evaluation results for segm:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.322
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.534
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.133
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.357
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.256
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 32.177 | 53.418 | 33.945 | 13.337 | 35.688 | 49.785 |
### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 41.932 | bicycle      | 15.383 | car            | 34.427 |  
| motorcycle    | 28.740 | airplane     | 40.015 | bus            | 59.467 |  
| train         | 56.676 | truck        | 28.831 | boat           | 18.890 |  
| traffic light | 22.146 | fire hydrant | 57.844 | stop sign      | 60.605 |  
| parking meter | 44.057 | bench        | 13.967 | bird           | 22.706 |  
| cat           | 63.775 | dog          | 51.504 | horse          | 35.187 |  
| sheep         | 36.760 | cow          | 40.086 | elephant       | 50.061 |  
| bear          | 61.556 | zebra        | 50.812 | giraffe        | 44.734 |  
| backpack      | 12.176 | umbrella     | 40.177 | handbag        | 9.282  |  
| tie           | 22.489 | suitcase     | 30.379 | frisbee        | 55.043 |  
| skis          | 0.881  | snowboard    | 19.096 | sports ball    | 38.598 |  
| kite          | 21.654 | baseball bat | 17.624 | baseball glove | 33.647 |  
| skateboard    | 24.517 | surfboard    | 24.108 | tennis racket  | 49.867 |  
| bottle        | 31.355 | wine glass   | 25.990 | cup            | 36.725 |  
| fork          | 10.795 | knife        | 6.731  | spoon          | 7.365  |  
| bowl          | 35.294 | banana       | 17.576 | apple          | 15.784 |  
| sandwich      | 30.196 | orange       | 26.440 | broccoli       | 19.866 |  
| carrot        | 15.627 | hot dog      | 23.439 | pizza          | 48.135 |  
| donut         | 40.300 | cake         | 31.572 | chair          | 14.696 |  
| couch         | 31.281 | potted plant | 19.112 | bed            | 27.226 |  
| dining table  | 13.542 | toilet       | 54.039 | tv             | 53.393 |  
| laptop        | 56.548 | mouse        | 53.070 | remote         | 17.741 |  
| keyboard      | 46.451 | cell phone   | 26.415 | microwave      | 53.344 |  
| oven          | 28.431 | toaster      | 46.385 | sink           | 28.281 |  
| refrigerator  | 49.245 | book         | 5.793  | clock          | 46.902 |  
| vase          | 30.139 | scissors     | 19.716 | teddy bear     | 40.341 |  
| hair drier    | 0.808  | toothbrush   | 8.368  |                |        |
