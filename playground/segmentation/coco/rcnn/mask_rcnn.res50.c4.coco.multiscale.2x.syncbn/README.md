# mask_rcnn.res50.c4.coco.multiscale.2x.syncbn  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.597
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.217
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.446
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.545
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.339
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.597
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.706
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 39.895 | 59.731 | 43.282 | 21.742 | 44.591 | 54.521 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 54.531 | bicycle      | 30.759 | car            | 42.423 |  
| motorcycle    | 43.027 | airplane     | 64.169 | bus            | 65.107 |  
| train         | 61.528 | truck        | 34.037 | boat           | 26.496 |  
| traffic light | 24.924 | fire hydrant | 66.717 | stop sign      | 63.625 |  
| parking meter | 46.919 | bench        | 23.274 | bird           | 33.805 |  
| cat           | 64.144 | dog          | 60.571 | horse          | 59.484 |  
| sheep         | 49.490 | cow          | 52.374 | elephant       | 63.186 |  
| bear          | 67.220 | zebra        | 67.904 | giraffe        | 68.124 |  
| backpack      | 15.275 | umbrella     | 40.400 | handbag        | 14.331 |  
| tie           | 29.351 | suitcase     | 34.220 | frisbee        | 63.734 |  
| skis          | 22.799 | snowboard    | 36.536 | sports ball    | 41.152 |  
| kite          | 37.064 | baseball bat | 26.983 | baseball glove | 33.367 |  
| skateboard    | 52.258 | surfboard    | 36.387 | tennis racket  | 46.647 |  
| bottle        | 36.510 | wine glass   | 34.306 | cup            | 40.083 |  
| fork          | 34.883 | knife        | 14.590 | spoon          | 15.757 |  
| bowl          | 41.625 | banana       | 23.644 | apple          | 20.090 |  
| sandwich      | 32.564 | orange       | 29.747 | broccoli       | 19.105 |  
| carrot        | 18.568 | hot dog      | 34.256 | pizza          | 51.924 |  
| donut         | 43.897 | cake         | 33.995 | chair          | 27.065 |  
| couch         | 41.250 | potted plant | 25.900 | bed            | 35.554 |  
| dining table  | 26.749 | toilet       | 59.021 | tv             | 55.197 |  
| laptop        | 59.545 | mouse        | 56.192 | remote         | 25.066 |  
| keyboard      | 49.571 | cell phone   | 33.305 | microwave      | 55.557 |  
| oven          | 31.403 | toaster      | 39.550 | sink           | 33.763 |  
| refrigerator  | 53.887 | book         | 12.775 | clock          | 50.179 |  
| vase          | 33.847 | scissors     | 26.659 | teddy bear     | 45.247 |  
| hair drier    | 5.521  | toothbrush   | 18.936 |                |        |
## Evaluation results for segm:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.564
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.148
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.380
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.298
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.457
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.644
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 34.534 | 56.357 | 36.755 | 14.757 | 38.033 | 52.577 |
### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 44.442 | bicycle      | 17.059 | car            | 38.180 |  
| motorcycle    | 33.023 | airplane     | 43.762 | bus            | 61.721 |  
| train         | 61.602 | truck        | 32.461 | boat           | 21.544 |  
| traffic light | 23.588 | fire hydrant | 62.341 | stop sign      | 61.554 |  
| parking meter | 46.443 | bench        | 15.813 | bird           | 25.280 |  
| cat           | 63.658 | dog          | 55.064 | horse          | 38.282 |  
| sheep         | 40.916 | cow          | 41.608 | elephant       | 55.121 |  
| bear          | 66.038 | zebra        | 53.417 | giraffe        | 45.392 |  
| backpack      | 13.991 | umbrella     | 44.133 | handbag        | 12.227 |  
| tie           | 24.416 | suitcase     | 35.945 | frisbee        | 59.474 |  
| skis          | 1.387  | snowboard    | 21.007 | sports ball    | 40.265 |  
| kite          | 23.886 | baseball bat | 20.171 | baseball glove | 34.365 |  
| skateboard    | 29.235 | surfboard    | 27.996 | tennis racket  | 51.826 |  
| bottle        | 34.082 | wine glass   | 29.013 | cup            | 39.434 |  
| fork          | 14.748 | knife        | 9.879  | spoon          | 8.882  |  
| bowl          | 37.631 | banana       | 19.139 | apple          | 18.704 |  
| sandwich      | 34.183 | orange       | 28.617 | broccoli       | 18.928 |  
| carrot        | 14.690 | hot dog      | 26.226 | pizza          | 49.367 |  
| donut         | 42.998 | cake         | 34.603 | chair          | 17.238 |  
| couch         | 32.897 | potted plant | 21.807 | bed            | 27.931 |  
| dining table  | 15.305 | toilet       | 56.937 | tv             | 56.799 |  
| laptop        | 57.746 | mouse        | 54.295 | remote         | 20.757 |  
| keyboard      | 47.572 | cell phone   | 30.095 | microwave      | 56.657 |  
| oven          | 29.402 | toaster      | 45.362 | sink           | 30.346 |  
| refrigerator  | 53.203 | book         | 6.706  | clock          | 50.014 |  
| vase          | 31.694 | scissors     | 17.555 | teddy bear     | 40.955 |  
| hair drier    | 1.363  | toothbrush   | 10.330 |                |        |
