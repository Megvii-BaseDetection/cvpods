# ssd.vgg16.coco.512size.expand_aug  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.290
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.480
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.110
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.337
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.259
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.388
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.406
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.147
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.467
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.604
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 28.995 | 48.034 | 30.914 | 10.981 | 33.709 | 44.860 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 37.952 | bicycle      | 20.819 | car            | 29.613 |  
| motorcycle    | 33.970 | airplane     | 52.928 | bus            | 56.786 |  
| train         | 55.267 | truck        | 28.630 | boat           | 15.251 |  
| traffic light | 12.025 | fire hydrant | 47.603 | stop sign      | 51.424 |  
| parking meter | 35.476 | bench        | 16.538 | bird           | 21.920 |  
| cat           | 54.706 | dog          | 51.152 | horse          | 45.410 |  
| sheep         | 37.394 | cow          | 40.689 | elephant       | 52.453 |  
| bear          | 60.862 | zebra        | 56.293 | giraffe        | 56.079 |  
| backpack      | 6.862  | umbrella     | 28.118 | handbag        | 5.725  |  
| tie           | 14.761 | suitcase     | 22.154 | frisbee        | 39.470 |  
| skis          | 11.274 | snowboard    | 17.564 | sports ball    | 23.791 |  
| kite          | 22.453 | baseball bat | 14.541 | baseball glove | 19.412 |  
| skateboard    | 36.434 | surfboard    | 23.138 | tennis racket  | 29.965 |  
| bottle        | 17.511 | wine glass   | 17.345 | cup            | 27.250 |  
| fork          | 14.714 | knife        | 5.587  | spoon          | 5.524  |  
| bowl          | 30.164 | banana       | 15.595 | apple          | 13.800 |  
| sandwich      | 31.273 | orange       | 22.921 | broccoli       | 18.037 |  
| carrot        | 15.282 | hot dog      | 24.065 | pizza          | 40.209 |  
| donut         | 31.966 | cake         | 25.255 | chair          | 17.258 |  
| couch         | 35.074 | potted plant | 16.378 | bed            | 38.154 |  
| dining table  | 22.691 | toilet       | 49.841 | tv             | 47.166 |  
| laptop        | 49.017 | mouse        | 44.073 | remote         | 12.764 |  
| keyboard      | 38.905 | cell phone   | 22.682 | microwave      | 46.678 |  
| oven          | 30.055 | toaster      | 13.925 | sink           | 24.699 |  
| refrigerator  | 40.709 | book         | 6.328  | clock          | 37.102 |  
| vase          | 21.273 | scissors     | 18.416 | teddy bear     | 36.488 |  
| hair drier    | 0.165  | toothbrush   | 6.353  |                |        |
