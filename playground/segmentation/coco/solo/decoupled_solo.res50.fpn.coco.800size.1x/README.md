# decoupled_solo.res50.fpn.coco.800size.1x  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.340
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.357
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.151
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.298
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.474
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.248
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.728
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 33.965 | 53.323 | 35.687 | 15.102 | 37.278 | 50.491 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 44.312 | bicycle      | 20.759 | car            | 32.239 |  
| motorcycle    | 34.954 | airplane     | 56.504 | bus            | 60.605 |  
| train         | 63.043 | truck        | 31.352 | boat           | 22.209 |  
| traffic light | 19.921 | fire hydrant | 59.669 | stop sign      | 56.947 |  
| parking meter | 39.624 | bench        | 16.966 | bird           | 24.875 |  
| cat           | 61.822 | dog          | 55.364 | horse          | 47.947 |  
| sheep         | 42.474 | cow          | 45.945 | elephant       | 57.663 |  
| bear          | 66.862 | zebra        | 63.168 | giraffe        | 60.661 |  
| backpack      | 8.609  | umbrella     | 27.143 | handbag        | 6.360  |  
| tie           | 24.247 | suitcase     | 31.598 | frisbee        | 56.569 |  
| skis          | 17.993 | snowboard    | 28.386 | sports ball    | 28.634 |  
| kite          | 24.142 | baseball bat | 16.308 | baseball glove | 25.232 |  
| skateboard    | 43.580 | surfboard    | 25.667 | tennis racket  | 29.834 |  
| bottle        | 26.398 | wine glass   | 21.246 | cup            | 31.890 |  
| fork          | 22.354 | knife        | 10.981 | spoon          | 8.502  |  
| bowl          | 32.996 | banana       | 20.191 | apple          | 16.618 |  
| sandwich      | 29.886 | orange       | 27.563 | broccoli       | 20.273 |  
| carrot        | 18.207 | hot dog      | 27.480 | pizza          | 43.825 |  
| donut         | 38.226 | cake         | 26.554 | chair          | 20.480 |  
| couch         | 38.308 | potted plant | 21.201 | bed            | 41.337 |  
| dining table  | 25.305 | toilet       | 55.547 | tv             | 50.135 |  
| laptop        | 49.949 | mouse        | 52.075 | remote         | 19.402 |  
| keyboard      | 46.314 | cell phone   | 26.870 | microwave      | 51.361 |  
| oven          | 33.473 | toaster      | 35.916 | sink           | 32.334 |  
| refrigerator  | 51.138 | book         | 8.037  | clock          | 44.151 |  
| vase          | 29.721 | scissors     | 24.272 | teddy bear     | 41.382 |  
| hair drier    | 1.922  | toothbrush   | 13.221 |                |        |


## Evaluation results for segm:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.537
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.359
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.129
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.364
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.289
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.473
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.231
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.676
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 33.723 | 53.737 | 35.885 | 12.933 | 36.425 | 52.141 |

### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 41.615 | bicycle      | 15.440 | car            | 34.083 |  
| motorcycle    | 29.170 | airplane     | 49.997 | bus            | 61.299 |  
| train         | 61.389 | truck        | 32.966 | boat           | 19.757 |  
| traffic light | 21.537 | fire hydrant | 59.601 | stop sign      | 57.362 |  
| parking meter | 40.601 | bench        | 14.452 | bird           | 26.694 |  
| cat           | 63.841 | dog          | 56.093 | horse          | 38.712 |  
| sheep         | 41.191 | cow          | 44.179 | elephant       | 56.967 |  
| bear          | 67.487 | zebra        | 57.401 | giraffe        | 52.499 |  
| backpack      | 10.533 | umbrella     | 42.328 | handbag        | 9.466  |  
| tie           | 26.638 | suitcase     | 34.137 | frisbee        | 58.216 |  
| skis          | 2.907  | snowboard    | 19.008 | sports ball    | 36.578 |  
| kite          | 25.418 | baseball bat | 22.423 | baseball glove | 30.648 |  
| skateboard    | 30.226 | surfboard    | 24.417 | tennis racket  | 48.840 |  
| bottle        | 27.398 | wine glass   | 24.512 | cup            | 34.496 |  
| fork          | 12.668 | knife        | 9.286  | spoon          | 7.063  |  
| bowl          | 33.613 | banana       | 18.027 | apple          | 17.329 |  
| sandwich      | 32.522 | orange       | 29.539 | broccoli       | 20.149 |  
| carrot        | 17.609 | hot dog      | 23.516 | pizza          | 44.746 |  
| donut         | 40.001 | cake         | 28.878 | chair          | 16.671 |  
| couch         | 34.424 | potted plant | 20.325 | bed            | 32.608 |  
| dining table  | 15.937 | toilet       | 58.713 | tv             | 53.619 |  
| laptop        | 54.757 | mouse        | 54.677 | remote         | 22.557 |  
| keyboard      | 47.522 | cell phone   | 29.386 | microwave      | 53.517 |  
| oven          | 31.381 | toaster      | 39.897 | sink           | 32.195 |  
| refrigerator  | 51.828 | book         | 4.747  | clock          | 47.271 |  
| vase          | 31.003 | scissors     | 20.239 | teddy bear     | 43.632 |  
| hair drier    | 1.683  | toothbrush   | 11.752 |                |        |
