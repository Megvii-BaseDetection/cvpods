# ssd.vgg16.coco.300size  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.236
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.243
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.050
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.423
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.226
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.336
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.063
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.381
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.579
```  
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |  
|:------:|:------:|:------:|:-----:|:------:|:------:|  
| 23.614 | 40.524 | 24.292 | 5.002 | 25.361 | 42.345 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 31.032 | bicycle      | 16.365 | car            | 19.400 |  
| motorcycle    | 27.383 | airplane     | 46.107 | bus            | 48.860 |  
| train         | 52.812 | truck        | 22.988 | boat           | 11.161 |  
| traffic light | 5.890  | fire hydrant | 42.466 | stop sign      | 45.998 |  
| parking meter | 29.586 | bench        | 14.165 | bird           | 16.648 |  
| cat           | 51.185 | dog          | 45.370 | horse          | 37.753 |  
| sheep         | 30.070 | cow          | 29.367 | elephant       | 46.127 |  
| bear          | 55.663 | zebra        | 47.029 | giraffe        | 48.190 |  
| backpack      | 4.499  | umbrella     | 21.610 | handbag        | 4.343  |  
| tie           | 10.655 | suitcase     | 17.223 | frisbee        | 27.217 |  
| skis          | 7.722  | snowboard    | 13.907 | sports ball    | 9.860  |  
| kite          | 15.334 | baseball bat | 10.505 | baseball glove | 12.618 |  
| skateboard    | 25.608 | surfboard    | 17.629 | tennis racket  | 23.958 |  
| bottle        | 11.748 | wine glass   | 12.287 | cup            | 19.076 |  
| fork          | 12.205 | knife        | 4.671  | spoon          | 3.832  |  
| bowl          | 24.585 | banana       | 13.058 | apple          | 9.984  |  
| sandwich      | 27.201 | orange       | 19.310 | broccoli       | 13.465 |  
| carrot        | 8.269  | hot dog      | 20.397 | pizza          | 36.973 |  
| donut         | 24.525 | cake         | 19.225 | chair          | 12.646 |  
| couch         | 32.344 | potted plant | 11.371 | bed            | 37.539 |  
| dining table  | 20.702 | toilet       | 48.029 | tv             | 41.559 |  
| laptop        | 41.470 | mouse        | 32.040 | remote         | 6.705  |  
| keyboard      | 32.049 | cell phone   | 16.953 | microwave      | 37.463 |  
| oven          | 24.780 | toaster      | 0.446  | sink           | 22.905 |  
| refrigerator  | 34.487 | book         | 3.867  | clock          | 28.056 |  
| vase          | 16.632 | scissors     | 22.645 | teddy bear     | 30.148 |  
| hair drier    | 2.475  | toothbrush   | 6.715  |                |        |
