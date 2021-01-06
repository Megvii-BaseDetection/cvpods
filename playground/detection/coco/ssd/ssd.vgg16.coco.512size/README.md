# ssd.vgg16.coco.512size  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.267
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.278
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.090
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.301
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.247
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.382
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.123
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.590
```  
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |  
|:------:|:------:|:------:|:-----:|:------:|:------:|  
| 26.674 | 45.307 | 27.805 | 9.027 | 30.086 | 43.819 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 35.888 | bicycle      | 19.438 | car            | 27.369 |  
| motorcycle    | 29.807 | airplane     | 49.456 | bus            | 52.498 |  
| train         | 52.572 | truck        | 24.876 | boat           | 13.819 |  
| traffic light | 10.740 | fire hydrant | 45.540 | stop sign      | 47.532 |  
| parking meter | 32.239 | bench        | 13.813 | bird           | 20.869 |  
| cat           | 54.522 | dog          | 48.419 | horse          | 42.323 |  
| sheep         | 33.591 | cow          | 39.046 | elephant       | 48.291 |  
| bear          | 58.223 | zebra        | 52.590 | giraffe        | 51.280 |  
| backpack      | 6.265  | umbrella     | 25.779 | handbag        | 5.379  |  
| tie           | 14.613 | suitcase     | 20.113 | frisbee        | 33.597 |  
| skis          | 11.054 | snowboard    | 17.557 | sports ball    | 21.274 |  
| kite          | 17.944 | baseball bat | 14.305 | baseball glove | 17.666 |  
| skateboard    | 32.290 | surfboard    | 22.441 | tennis racket  | 30.486 |  
| bottle        | 16.185 | wine glass   | 16.814 | cup            | 25.947 |  
| fork          | 13.577 | knife        | 5.720  | spoon          | 4.365  |  
| bowl          | 27.071 | banana       | 14.866 | apple          | 11.597 |  
| sandwich      | 26.970 | orange       | 20.908 | broccoli       | 15.733 |  
| carrot        | 10.543 | hot dog      | 20.561 | pizza          | 38.688 |  
| donut         | 27.110 | cake         | 22.614 | chair          | 15.166 |  
| couch         | 32.716 | potted plant | 14.641 | bed            | 32.858 |  
| dining table  | 20.032 | toilet       | 49.132 | tv             | 44.213 |  
| laptop        | 46.124 | mouse        | 40.343 | remote         | 12.778 |  
| keyboard      | 34.995 | cell phone   | 20.666 | microwave      | 43.445 |  
| oven          | 25.971 | toaster      | 6.764  | sink           | 21.107 |  
| refrigerator  | 36.397 | book         | 5.312  | clock          | 36.006 |  
| vase          | 18.171 | scissors     | 26.240 | teddy bear     | 29.902 |  
| hair drier    | 0.990  | toothbrush   | 5.222  |                |        |
