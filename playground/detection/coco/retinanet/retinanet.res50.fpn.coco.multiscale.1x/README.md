# retinanet.res50.fpn.coco.multiscale.1x  

seed: 54373550

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.561
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.395
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.407
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.541
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.355
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 36.972 | 56.094 | 39.523 | 22.692 | 40.713 | 48.540 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 50.773 | bicycle      | 27.125 | car            | 39.880 |  
| motorcycle    | 40.405 | airplane     | 63.344 | bus            | 63.718 |  
| train         | 59.472 | truck        | 33.432 | boat           | 23.513 |  
| traffic light | 25.051 | fire hydrant | 63.709 | stop sign      | 62.338 |  
| parking meter | 43.618 | bench        | 20.839 | bird           | 32.856 |  
| cat           | 64.382 | dog          | 60.487 | horse          | 51.448 |  
| sheep         | 46.863 | cow          | 49.293 | elephant       | 56.725 |  
| bear          | 67.596 | zebra        | 64.431 | giraffe        | 62.447 |  
| backpack      | 12.816 | umbrella     | 33.575 | handbag        | 11.732 |  
| tie           | 26.924 | suitcase     | 32.361 | frisbee        | 61.239 |  
| skis          | 16.860 | snowboard    | 18.176 | sports ball    | 43.461 |  
| kite          | 36.440 | baseball bat | 23.016 | baseball glove | 30.666 |  
| skateboard    | 46.296 | surfboard    | 30.590 | tennis racket  | 43.959 |  
| bottle        | 34.105 | wine glass   | 32.470 | cup            | 38.667 |  
| fork          | 21.774 | knife        | 10.702 | spoon          | 8.498  |  
| bowl          | 38.039 | banana       | 21.952 | apple          | 18.406 |  
| sandwich      | 29.000 | orange       | 27.158 | broccoli       | 21.329 |  
| carrot        | 19.073 | hot dog      | 25.912 | pizza          | 47.289 |  
| donut         | 39.409 | cake         | 30.148 | chair          | 23.320 |  
| couch         | 38.598 | potted plant | 23.030 | bed            | 41.347 |  
| dining table  | 24.819 | toilet       | 55.766 | tv             | 53.597 |  
| laptop        | 54.526 | mouse        | 59.451 | remote         | 24.147 |  
| keyboard      | 43.568 | cell phone   | 32.373 | microwave      | 55.890 |  
| oven          | 31.385 | toaster      | 26.014 | sink           | 32.295 |  
| refrigerator  | 49.247 | book         | 12.279 | clock          | 48.616 |  
| vase          | 35.170 | scissors     | 24.159 | teddy bear     | 42.737 |  
| hair drier    | 5.050  | toothbrush   | 14.578 |                |        |
