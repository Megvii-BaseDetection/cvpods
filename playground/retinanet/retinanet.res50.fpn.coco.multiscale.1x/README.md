# retinanet.res50.fpn.coco.multiscale.1x  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.365
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.562
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.393
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.314
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.534
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.348
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.683
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 36.454 | 56.242 | 39.328 | 21.898 | 40.478 | 47.747 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 49.847 | bicycle      | 27.800 | car            | 39.543 |  
| motorcycle    | 39.535 | airplane     | 60.788 | bus            | 62.082 |  
| train         | 57.897 | truck        | 32.374 | boat           | 23.642 |  
| traffic light | 23.785 | fire hydrant | 61.860 | stop sign      | 62.912 |  
| parking meter | 42.134 | bench        | 20.323 | bird           | 32.035 |  
| cat           | 62.643 | dog          | 59.460 | horse          | 51.436 |  
| sheep         | 45.442 | cow          | 49.267 | elephant       | 55.752 |  
| bear          | 69.257 | zebra        | 62.308 | giraffe        | 62.423 |  
| backpack      | 12.308 | umbrella     | 32.660 | handbag        | 11.692 |  
| tie           | 26.095 | suitcase     | 29.544 | frisbee        | 62.662 |  
| skis          | 18.217 | snowboard    | 22.185 | sports ball    | 43.453 |  
| kite          | 37.063 | baseball bat | 21.984 | baseball glove | 31.171 |  
| skateboard    | 48.046 | surfboard    | 30.663 | tennis racket  | 44.667 |  
| bottle        | 32.999 | wine glass   | 30.731 | cup            | 37.611 |  
| fork          | 22.843 | knife        | 9.941  | spoon          | 10.597 |  
| bowl          | 37.554 | banana       | 21.658 | apple          | 16.915 |  
| sandwich      | 28.278 | orange       | 27.994 | broccoli       | 20.984 |  
| carrot        | 19.075 | hot dog      | 27.483 | pizza          | 46.153 |  
| donut         | 39.205 | cake         | 30.042 | chair          | 23.022 |  
| couch         | 36.458 | potted plant | 23.638 | bed            | 39.539 |  
| dining table  | 24.671 | toilet       | 54.413 | tv             | 53.012 |  
| laptop        | 52.960 | mouse        | 59.783 | remote         | 24.399 |  
| keyboard      | 42.980 | cell phone   | 32.588 | microwave      | 53.939 |  
| oven          | 31.974 | toaster      | 16.414 | sink           | 31.876 |  
| refrigerator  | 46.588 | book         | 11.818 | clock          | 48.778 |  
| vase          | 34.466 | scissors     | 25.628 | teddy bear     | 45.000 |  
| hair drier    | 0.428  | toothbrush   | 14.942 |                |        |
