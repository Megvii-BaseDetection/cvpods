# decoupled_solo.res50.fpn.coco.multiscale.3x  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.359
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.564
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.378
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.172
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.395
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.518
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.483
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.716
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 35.884 | 56.359 | 37.786 | 17.179 | 39.519 | 51.815 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 46.161 | bicycle      | 23.728 | car            | 35.031 |  
| motorcycle    | 37.016 | airplane     | 58.142 | bus            | 63.988 |  
| train         | 66.634 | truck        | 34.586 | boat           | 22.596 |  
| traffic light | 21.677 | fire hydrant | 58.430 | stop sign      | 59.682 |  
| parking meter | 42.665 | bench        | 20.252 | bird           | 26.941 |  
| cat           | 61.748 | dog          | 55.550 | horse          | 52.698 |  
| sheep         | 46.369 | cow          | 47.925 | elephant       | 60.585 |  
| bear          | 66.912 | zebra        | 63.340 | giraffe        | 60.490 |  
| backpack      | 11.078 | umbrella     | 29.921 | handbag        | 9.112  |  
| tie           | 25.975 | suitcase     | 35.620 | frisbee        | 56.784 |  
| skis          | 20.415 | snowboard    | 31.781 | sports ball    | 31.366 |  
| kite          | 25.241 | baseball bat | 15.539 | baseball glove | 29.983 |  
| skateboard    | 46.760 | surfboard    | 28.217 | tennis racket  | 35.160 |  
| bottle        | 28.702 | wine glass   | 26.495 | cup            | 34.499 |  
| fork          | 26.362 | knife        | 13.991 | spoon          | 11.322 |  
| bowl          | 33.407 | banana       | 19.586 | apple          | 16.551 |  
| sandwich      | 29.034 | orange       | 26.314 | broccoli       | 22.598 |  
| carrot        | 20.474 | hot dog      | 32.258 | pizza          | 42.975 |  
| donut         | 43.244 | cake         | 32.420 | chair          | 23.013 |  
| couch         | 41.145 | potted plant | 22.632 | bed            | 42.083 |  
| dining table  | 25.569 | toilet       | 57.831 | tv             | 52.658 |  
| laptop        | 51.851 | mouse        | 54.202 | remote         | 21.881 |  
| keyboard      | 47.248 | cell phone   | 28.851 | microwave      | 53.509 |  
| oven          | 32.834 | toaster      | 26.410 | sink           | 32.636 |  
| refrigerator  | 56.054 | book         | 8.720  | clock          | 46.525 |  
| vase          | 33.252 | scissors     | 24.521 | teddy bear     | 42.007 |  
| hair drier    | 3.010  | toothbrush   | 15.940 |                |        |


## Evaluation results for segm:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.356
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.568
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.378
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.273
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.663
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 35.646 | 56.843 | 37.810 | 15.395 | 38.721 | 53.097 |

### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 43.445 | bicycle      | 17.523 | car            | 36.923 |  
| motorcycle    | 31.575 | airplane     | 50.927 | bus            | 63.690 |  
| train         | 65.330 | truck        | 36.292 | boat           | 22.701 |  
| traffic light | 24.272 | fire hydrant | 59.947 | stop sign      | 60.072 |  
| parking meter | 45.352 | bench        | 16.999 | bird           | 27.389 |  
| cat           | 63.576 | dog          | 54.889 | horse          | 43.243 |  
| sheep         | 44.468 | cow          | 46.658 | elephant       | 59.248 |  
| bear          | 67.733 | zebra        | 59.708 | giraffe        | 51.140 |  
| backpack      | 12.937 | umbrella     | 44.359 | handbag        | 13.403 |  
| tie           | 28.716 | suitcase     | 37.768 | frisbee        | 61.164 |  
| skis          | 4.581  | snowboard    | 22.132 | sports ball    | 39.072 |  
| kite          | 26.902 | baseball bat | 22.017 | baseball glove | 37.776 |  
| skateboard    | 32.683 | surfboard    | 27.543 | tennis racket  | 51.476 |  
| bottle        | 29.995 | wine glass   | 28.745 | cup            | 37.594 |  
| fork          | 15.188 | knife        | 11.872 | spoon          | 9.529  |  
| bowl          | 32.625 | banana       | 16.573 | apple          | 16.943 |  
| sandwich      | 31.238 | orange       | 26.891 | broccoli       | 21.985 |  
| carrot        | 20.275 | hot dog      | 26.646 | pizza          | 44.252 |  
| donut         | 45.592 | cake         | 34.067 | chair          | 18.823 |  
| couch         | 36.938 | potted plant | 21.638 | bed            | 32.073 |  
| dining table  | 14.328 | toilet       | 60.556 | tv             | 55.484 |  
| laptop        | 56.508 | mouse        | 59.343 | remote         | 25.327 |  
| keyboard      | 49.695 | cell phone   | 32.388 | microwave      | 54.044 |  
| oven          | 31.844 | toaster      | 32.920 | sink           | 33.938 |  
| refrigerator  | 53.672 | book         | 5.832  | clock          | 49.754 |  
| vase          | 34.708 | scissors     | 20.443 | teddy bear     | 42.694 |  
| hair drier    | 9.169  | toothbrush   | 13.937 |                |        |
