# solo.res50.fpn.coco.800size.1x  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.331
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.525
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.153
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.290
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.494
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.248
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.713
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 33.134 | 52.489 | 34.519 | 15.268 | 36.542 | 48.499 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 43.031 | bicycle      | 21.393 | car            | 30.503 |  
| motorcycle    | 34.806 | airplane     | 55.751 | bus            | 60.839 |  
| train         | 64.255 | truck        | 30.948 | boat           | 20.598 |  
| traffic light | 19.593 | fire hydrant | 57.894 | stop sign      | 55.576 |  
| parking meter | 38.376 | bench        | 17.510 | bird           | 24.073 |  
| cat           | 60.970 | dog          | 54.557 | horse          | 46.544 |  
| sheep         | 43.776 | cow          | 45.317 | elephant       | 57.727 |  
| bear          | 69.590 | zebra        | 58.896 | giraffe        | 59.016 |  
| backpack      | 8.593  | umbrella     | 26.206 | handbag        | 6.794  |  
| tie           | 23.012 | suitcase     | 27.446 | frisbee        | 53.608 |  
| skis          | 17.878 | snowboard    | 27.610 | sports ball    | 27.889 |  
| kite          | 24.538 | baseball bat | 15.936 | baseball glove | 26.549 |  
| skateboard    | 45.720 | surfboard    | 25.484 | tennis racket  | 30.853 |  
| bottle        | 25.546 | wine glass   | 20.076 | cup            | 31.304 |  
| fork          | 22.266 | knife        | 9.260  | spoon          | 7.318  |  
| bowl          | 33.379 | banana       | 20.666 | apple          | 15.825 |  
| sandwich      | 31.181 | orange       | 26.534 | broccoli       | 19.798 |  
| carrot        | 16.763 | hot dog      | 29.161 | pizza          | 42.704 |  
| donut         | 37.545 | cake         | 27.579 | chair          | 19.796 |  
| couch         | 39.387 | potted plant | 20.310 | bed            | 41.375 |  
| dining table  | 24.585 | toilet       | 57.775 | tv             | 50.025 |  
| laptop        | 51.921 | mouse        | 49.554 | remote         | 16.960 |  
| keyboard      | 42.109 | cell phone   | 26.440 | microwave      | 51.839 |  
| oven          | 32.208 | toaster      | 12.366 | sink           | 32.726 |  
| refrigerator  | 50.297 | book         | 6.912  | clock          | 43.921 |  
| vase          | 28.592 | scissors     | 21.688 | teddy bear     | 41.174 |  
| hair drier    | 0.951  | toothbrush   | 11.225 |                |        |


## Evaluation results for segm:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.327
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.530
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.125
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.357
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.280
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.230
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 32.733 | 53.050 | 34.399 | 12.515 | 35.709 | 50.232 |

### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 40.021 | bicycle      | 15.005 | car            | 32.205 |  
| motorcycle    | 28.914 | airplane     | 49.999 | bus            | 61.029 |  
| train         | 61.662 | truck        | 33.317 | boat           | 18.835 |  
| traffic light | 20.661 | fire hydrant | 57.898 | stop sign      | 55.227 |  
| parking meter | 37.713 | bench        | 14.039 | bird           | 25.682 |  
| cat           | 64.077 | dog          | 54.737 | horse          | 38.215 |  
| sheep         | 41.283 | cow          | 44.340 | elephant       | 55.988 |  
| bear          | 70.791 | zebra        | 55.181 | giraffe        | 49.507 |  
| backpack      | 10.909 | umbrella     | 41.618 | handbag        | 9.840  |  
| tie           | 26.266 | suitcase     | 30.438 | frisbee        | 57.709 |  
| skis          | 2.990  | snowboard    | 18.473 | sports ball    | 36.227 |  
| kite          | 25.675 | baseball bat | 20.175 | baseball glove | 32.060 |  
| skateboard    | 30.754 | surfboard    | 24.927 | tennis racket  | 47.105 |  
| bottle        | 26.220 | wine glass   | 23.871 | cup            | 34.188 |  
| fork          | 11.733 | knife        | 8.742  | spoon          | 6.443  |  
| bowl          | 33.315 | banana       | 17.350 | apple          | 16.553 |  
| sandwich      | 32.947 | orange       | 27.160 | broccoli       | 19.884 |  
| carrot        | 16.533 | hot dog      | 23.919 | pizza          | 43.915 |  
| donut         | 39.780 | cake         | 29.128 | chair          | 15.426 |  
| couch         | 34.270 | potted plant | 19.240 | bed            | 31.176 |  
| dining table  | 15.025 | toilet       | 59.337 | tv             | 52.939 |  
| laptop        | 55.399 | mouse        | 53.162 | remote         | 21.077 |  
| keyboard      | 44.119 | cell phone   | 28.759 | microwave      | 51.726 |  
| oven          | 29.533 | toaster      | 18.334 | sink           | 32.817 |  
| refrigerator  | 49.851 | book         | 4.355  | clock          | 46.139 |  
| vase          | 29.908 | scissors     | 18.726 | teddy bear     | 41.730 |  
| hair drier    | 2.943  | toothbrush   | 9.469  |                |        |
