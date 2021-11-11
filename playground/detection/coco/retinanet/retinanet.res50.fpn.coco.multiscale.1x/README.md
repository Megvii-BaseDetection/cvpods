# retinanet.res50.fpn.coco.multiscale.1x  

seed: 44564257

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.567
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.232
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.548
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.367
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 37.264 | 56.666 | 39.861 | 23.209 | 41.492 | 47.683 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 51.457 | bicycle      | 30.013 | car            | 40.911 |  
| motorcycle    | 40.632 | airplane     | 62.267 | bus            | 64.274 |  
| train         | 57.423 | truck        | 33.324 | boat           | 23.266 |  
| traffic light | 24.660 | fire hydrant | 61.616 | stop sign      | 63.538 |  
| parking meter | 42.614 | bench        | 20.373 | bird           | 33.594 |  
| cat           | 63.453 | dog          | 59.041 | horse          | 52.662 |  
| sheep         | 47.561 | cow          | 50.292 | elephant       | 57.373 |  
| bear          | 67.403 | zebra        | 64.911 | giraffe        | 63.306 |  
| backpack      | 13.978 | umbrella     | 34.438 | handbag        | 13.146 |  
| tie           | 26.924 | suitcase     | 34.480 | frisbee        | 62.618 |  
| skis          | 16.099 | snowboard    | 22.448 | sports ball    | 42.871 |  
| kite          | 37.310 | baseball bat | 21.322 | baseball glove | 30.943 |  
| skateboard    | 45.886 | surfboard    | 30.276 | tennis racket  | 43.892 |  
| bottle        | 35.031 | wine glass   | 35.021 | cup            | 39.936 |  
| fork          | 21.902 | knife        | 11.794 | spoon          | 10.065 |  
| bowl          | 39.587 | banana       | 23.173 | apple          | 17.891 |  
| sandwich      | 30.569 | orange       | 29.368 | broccoli       | 21.506 |  
| carrot        | 18.314 | hot dog      | 26.927 | pizza          | 47.359 |  
| donut         | 41.433 | cake         | 33.014 | chair          | 25.245 |  
| couch         | 37.060 | potted plant | 25.304 | bed            | 39.847 |  
| dining table  | 25.716 | toilet       | 55.300 | tv             | 53.393 |  
| laptop        | 55.369 | mouse        | 59.776 | remote         | 25.094 |  
| keyboard      | 42.249 | cell phone   | 32.547 | microwave      | 56.187 |  
| oven          | 32.518 | toaster      | 23.279 | sink           | 32.723 |  
| refrigerator  | 48.203 | book         | 12.131 | clock          | 48.602 |  
| vase          | 35.384 | scissors     | 23.036 | teddy bear     | 43.832 |  
| hair drier    | 0.586  | toothbrush   | 12.286 |                |        |
