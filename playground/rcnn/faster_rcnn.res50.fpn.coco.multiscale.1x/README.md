# faster_rcnn.res50.fpn.coco.multiscale.1x  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.381
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.589
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.410
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.225
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.489
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.315
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.655
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 38.057 | 58.911 | 40.977 | 22.452 | 41.475 | 48.905 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 52.845 | bicycle      | 28.890 | car            | 42.291 |  
| motorcycle    | 40.005 | airplane     | 59.998 | bus            | 60.964 |  
| train         | 55.034 | truck        | 31.727 | boat           | 25.279 |  
| traffic light | 26.619 | fire hydrant | 64.046 | stop sign      | 65.040 |  
| parking meter | 43.666 | bench        | 20.882 | bird           | 34.395 |  
| cat           | 60.255 | dog          | 55.087 | horse          | 53.924 |  
| sheep         | 47.932 | cow          | 51.431 | elephant       | 58.283 |  
| bear          | 66.836 | zebra        | 62.944 | giraffe        | 63.190 |  
| backpack      | 13.190 | umbrella     | 34.237 | handbag        | 11.296 |  
| tie           | 30.755 | suitcase     | 33.169 | frisbee        | 61.428 |  
| skis          | 20.723 | snowboard    | 31.380 | sports ball    | 46.417 |  
| kite          | 40.287 | baseball bat | 25.261 | baseball glove | 34.025 |  
| skateboard    | 46.695 | surfboard    | 34.559 | tennis racket  | 43.375 |  
| bottle        | 37.374 | wine glass   | 33.280 | cup            | 38.928 |  
| fork          | 26.670 | knife        | 14.192 | spoon          | 12.417 |  
| bowl          | 40.936 | banana       | 22.176 | apple          | 18.623 |  
| sandwich      | 32.307 | orange       | 28.723 | broccoli       | 22.731 |  
| carrot        | 20.228 | hot dog      | 27.182 | pizza          | 49.065 |  
| donut         | 40.512 | cake         | 32.857 | chair          | 24.171 |  
| couch         | 37.348 | potted plant | 23.652 | bed            | 36.613 |  
| dining table  | 25.366 | toilet       | 54.125 | tv             | 51.871 |  
| laptop        | 55.955 | mouse        | 61.394 | remote         | 26.060 |  
| keyboard      | 47.388 | cell phone   | 33.664 | microwave      | 53.486 |  
| oven          | 29.369 | toaster      | 41.430 | sink           | 33.269 |  
| refrigerator  | 49.589 | book         | 13.998 | clock          | 49.126 |  
| vase          | 36.455 | scissors     | 22.927 | teddy bear     | 42.754 |  
| hair drier    | 0.495  | toothbrush   | 17.498 |                |        |
