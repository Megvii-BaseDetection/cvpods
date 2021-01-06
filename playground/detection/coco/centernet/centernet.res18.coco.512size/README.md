# centernet.res18.coco.512size  

seed: 22335028

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.298
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.466
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.315
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.106
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.332
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.272
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.432
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.451
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.491
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 29.835 | 46.648 | 31.547 | 10.626 | 33.151 | 45.174 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 42.317 | bicycle      | 20.338 | car            | 27.891 |  
| motorcycle    | 29.542 | airplane     | 53.908 | bus            | 53.012 |  
| train         | 56.144 | truck        | 24.419 | boat           | 15.006 |  
| traffic light | 14.682 | fire hydrant | 54.600 | stop sign      | 52.863 |  
| parking meter | 37.097 | bench        | 16.277 | bird           | 22.490 |  
| cat           | 56.954 | dog          | 51.676 | horse          | 48.218 |  
| sheep         | 39.688 | cow          | 44.122 | elephant       | 54.674 |  
| bear          | 69.742 | zebra        | 57.988 | giraffe        | 62.251 |  
| backpack      | 7.181  | umbrella     | 28.612 | handbag        | 6.241  |  
| tie           | 18.312 | suitcase     | 25.269 | frisbee        | 44.562 |  
| skis          | 13.368 | snowboard    | 19.790 | sports ball    | 26.428 |  
| kite          | 27.945 | baseball bat | 16.162 | baseball glove | 21.647 |  
| skateboard    | 38.350 | surfboard    | 24.245 | tennis racket  | 33.997 |  
| bottle        | 21.809 | wine glass   | 21.605 | cup            | 26.694 |  
| fork          | 17.511 | knife        | 6.253  | spoon          | 5.149  |  
| bowl          | 27.679 | banana       | 17.829 | apple          | 12.609 |  
| sandwich      | 27.050 | orange       | 23.757 | broccoli       | 14.589 |  
| carrot        | 14.402 | hot dog      | 22.030 | pizza          | 40.817 |  
| donut         | 33.925 | cake         | 27.445 | chair          | 17.751 |  
| couch         | 39.024 | potted plant | 17.977 | bed            | 34.603 |  
| dining table  | 18.031 | toilet       | 54.822 | tv             | 45.718 |  
| laptop        | 46.794 | mouse        | 43.198 | remote         | 9.584  |  
| keyboard      | 40.009 | cell phone   | 20.973 | microwave      | 47.592 |  
| oven          | 29.030 | toaster      | 5.764  | sink           | 26.762 |  
| refrigerator  | 44.178 | book         | 4.012  | clock          | 38.651 |  
| vase          | 24.807 | scissors     | 16.805 | teddy bear     | 34.645 |  
| hair drier    | 0.062  | toothbrush   | 6.815  |                |        |
