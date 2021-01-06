# coco.res50.fpn.1x.ms_train  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.409
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.221
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.407
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.313
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.497
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.335
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.658
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 37.647 | 58.616 | 40.914 | 22.090 | 40.714 | 48.875 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 52.654 | bicycle      | 28.865 | car            | 41.988 |  
| motorcycle    | 39.117 | airplane     | 60.542 | bus            | 61.192 |  
| train         | 58.431 | truck        | 31.451 | boat           | 24.459 |  
| traffic light | 27.102 | fire hydrant | 63.804 | stop sign      | 63.911 |  
| parking meter | 46.855 | bench        | 21.608 | bird           | 33.029 |  
| cat           | 60.802 | dog          | 54.394 | horse          | 51.828 |  
| sheep         | 45.956 | cow          | 50.923 | elephant       | 58.783 |  
| bear          | 65.185 | zebra        | 62.686 | giraffe        | 63.487 |  
| backpack      | 12.117 | umbrella     | 35.205 | handbag        | 10.202 |  
| tie           | 29.210 | suitcase     | 32.014 | frisbee        | 60.702 |  
| skis          | 18.879 | snowboard    | 28.623 | sports ball    | 45.355 |  
| kite          | 40.612 | baseball bat | 23.255 | baseball glove | 32.601 |  
| skateboard    | 44.649 | surfboard    | 33.105 | tennis racket  | 41.030 |  
| bottle        | 37.145 | wine glass   | 32.713 | cup            | 38.855 |  
| fork          | 27.013 | knife        | 13.890 | spoon          | 12.648 |  
| bowl          | 40.538 | banana       | 23.015 | apple          | 18.098 |  
| sandwich      | 31.978 | orange       | 27.855 | broccoli       | 21.093 |  
| carrot        | 20.329 | hot dog      | 26.530 | pizza          | 48.678 |  
| donut         | 40.337 | cake         | 32.366 | chair          | 23.621 |  
| couch         | 38.655 | potted plant | 22.418 | bed            | 38.440 |  
| dining table  | 25.466 | toilet       | 54.289 | tv             | 53.073 |  
| laptop        | 54.872 | mouse        | 59.981 | remote         | 24.036 |  
| keyboard      | 48.687 | cell phone   | 32.675 | microwave      | 53.099 |  
| oven          | 29.399 | toaster      | 39.418 | sink           | 33.280 |  
| refrigerator  | 48.695 | book         | 13.556 | clock          | 48.687 |  
| vase          | 36.536 | scissors     | 23.878 | teddy bear     | 41.987 |  
| hair drier    | 0.360  | toothbrush   | 16.975 |                |        |
## Evaluation results for segm:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.557
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.165
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.296
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.481
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.628
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 34.665 | 55.748 | 37.145 | 16.522 | 36.936 | 49.787 |
### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 45.278 | bicycle      | 16.458 | car            | 38.600 |  
| motorcycle    | 29.089 | airplane     | 48.590 | bus            | 60.496 |  
| train         | 58.769 | truck        | 30.693 | boat           | 22.069 |  
| traffic light | 26.159 | fire hydrant | 61.416 | stop sign      | 64.367 |  
| parking meter | 48.496 | bench        | 16.062 | bird           | 27.183 |  
| cat           | 65.137 | dog          | 54.576 | horse          | 37.930 |  
| sheep         | 39.839 | cow          | 44.332 | elephant       | 54.421 |  
| bear          | 65.163 | zebra        | 54.046 | giraffe        | 47.524 |  
| backpack      | 12.287 | umbrella     | 42.803 | handbag        | 11.805 |  
| tie           | 27.563 | suitcase     | 34.179 | frisbee        | 59.490 |  
| skis          | 2.211  | snowboard    | 19.495 | sports ball    | 44.703 |  
| kite          | 29.785 | baseball bat | 20.743 | baseball glove | 36.685 |  
| skateboard    | 25.973 | surfboard    | 26.715 | tennis racket  | 50.624 |  
| bottle        | 36.029 | wine glass   | 28.471 | cup            | 39.266 |  
| fork          | 11.218 | knife        | 9.001  | spoon          | 8.002  |  
| bowl          | 38.028 | banana       | 17.974 | apple          | 17.789 |  
| sandwich      | 34.054 | orange       | 28.216 | broccoli       | 20.475 |  
| carrot        | 17.894 | hot dog      | 19.277 | pizza          | 48.406 |  
| donut         | 41.390 | cake         | 33.420 | chair          | 15.629 |  
| couch         | 33.721 | potted plant | 20.410 | bed            | 31.675 |  
| dining table  | 13.843 | toilet       | 54.710 | tv             | 55.870 |  
| laptop        | 56.330 | mouse        | 59.908 | remote         | 24.350 |  
| keyboard      | 48.738 | cell phone   | 31.733 | microwave      | 55.757 |  
| oven          | 29.097 | toaster      | 40.746 | sink           | 31.468 |  
| refrigerator  | 51.754 | book         | 8.707  | clock          | 49.418 |  
| vase          | 36.362 | scissors     | 19.759 | teddy bear     | 40.222 |  
| hair drier    | 1.635  | toothbrush   | 10.681 |                |        |
