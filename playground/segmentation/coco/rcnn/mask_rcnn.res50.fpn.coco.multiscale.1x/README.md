# mask_rcnn.res50.fpn.coco.multiscale.1x  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.591
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.419
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.504
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 38.461 | 59.133 | 41.932 | 22.353 | 41.636 | 50.202 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 53.573 | bicycle      | 29.079 | car            | 42.832 |  
| motorcycle    | 39.602 | airplane     | 59.804 | bus            | 61.190 |  
| train         | 59.406 | truck        | 30.464 | boat           | 25.438 |  
| traffic light | 28.088 | fire hydrant | 64.168 | stop sign      | 66.202 |  
| parking meter | 43.773 | bench        | 22.335 | bird           | 34.497 |  
| cat           | 61.929 | dog          | 56.598 | horse          | 53.921 |  
| sheep         | 50.055 | cow          | 51.910 | elephant       | 59.148 |  
| bear          | 66.557 | zebra        | 63.723 | giraffe        | 63.847 |  
| backpack      | 13.282 | umbrella     | 34.967 | handbag        | 11.689 |  
| tie           | 30.382 | suitcase     | 34.803 | frisbee        | 61.409 |  
| skis          | 19.480 | snowboard    | 30.919 | sports ball    | 47.221 |  
| kite          | 40.335 | baseball bat | 24.971 | baseball glove | 34.223 |  
| skateboard    | 46.175 | surfboard    | 35.354 | tennis racket  | 43.658 |  
| bottle        | 37.230 | wine glass   | 33.918 | cup            | 39.908 |  
| fork          | 28.697 | knife        | 15.329 | spoon          | 13.458 |  
| bowl          | 40.112 | banana       | 23.274 | apple          | 16.859 |  
| sandwich      | 31.624 | orange       | 28.822 | broccoli       | 22.577 |  
| carrot        | 19.614 | hot dog      | 29.029 | pizza          | 49.336 |  
| donut         | 42.115 | cake         | 35.283 | chair          | 25.181 |  
| couch         | 37.147 | potted plant | 23.180 | bed            | 41.162 |  
| dining table  | 25.915 | toilet       | 55.818 | tv             | 52.653 |  
| laptop        | 55.481 | mouse        | 60.167 | remote         | 26.521 |  
| keyboard      | 47.095 | cell phone   | 31.734 | microwave      | 54.203 |  
| oven          | 31.582 | toaster      | 37.093 | sink           | 34.371 |  
| refrigerator  | 52.625 | book         | 14.128 | clock          | 47.444 |  
| vase          | 35.435 | scissors     | 21.879 | teddy bear     | 42.493 |  
| hair drier    | 1.980  | toothbrush   | 17.407 |                |        |
## Evaluation results for segm:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.562
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.376
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.169
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.484
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.298
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.628
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 35.162 | 56.245 | 37.576 | 16.898 | 37.393 | 50.791 |
### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 45.943 | bicycle      | 17.014 | car            | 38.955 |  
| motorcycle    | 29.985 | airplane     | 48.480 | bus            | 61.165 |  
| train         | 58.547 | truck        | 30.587 | boat           | 22.112 |  
| traffic light | 26.580 | fire hydrant | 62.079 | stop sign      | 66.001 |  
| parking meter | 45.577 | bench        | 16.171 | bird           | 29.034 |  
| cat           | 64.267 | dog          | 54.956 | horse          | 39.352 |  
| sheep         | 42.683 | cow          | 45.335 | elephant       | 54.268 |  
| bear          | 64.953 | zebra        | 55.963 | giraffe        | 48.881 |  
| backpack      | 13.595 | umbrella     | 41.690 | handbag        | 13.035 |  
| tie           | 29.413 | suitcase     | 36.700 | frisbee        | 60.188 |  
| skis          | 1.877  | snowboard    | 20.721 | sports ball    | 45.851 |  
| kite          | 29.987 | baseball bat | 21.478 | baseball glove | 36.614 |  
| skateboard    | 28.397 | surfboard    | 29.654 | tennis racket  | 51.667 |  
| bottle        | 35.682 | wine glass   | 29.537 | cup            | 40.038 |  
| fork          | 12.735 | knife        | 9.381  | spoon          | 7.960  |  
| bowl          | 37.900 | banana       | 18.443 | apple          | 16.206 |  
| sandwich      | 33.728 | orange       | 28.994 | broccoli       | 21.426 |  
| carrot        | 17.036 | hot dog      | 21.447 | pizza          | 49.144 |  
| donut         | 43.079 | cake         | 35.765 | chair          | 16.367 |  
| couch         | 31.980 | potted plant | 20.879 | bed            | 32.914 |  
| dining table  | 15.112 | toilet       | 55.809 | tv             | 55.202 |  
| laptop        | 56.878 | mouse        | 61.460 | remote         | 24.976 |  
| keyboard      | 47.449 | cell phone   | 32.238 | microwave      | 54.530 |  
| oven          | 29.236 | toaster      | 37.941 | sink           | 32.500 |  
| refrigerator  | 54.466 | book         | 9.413  | clock          | 48.653 |  
| vase          | 35.956 | scissors     | 16.939 | teddy bear     | 41.349 |  
| hair drier    | 0.705  | toothbrush   | 11.791 |                |        |
