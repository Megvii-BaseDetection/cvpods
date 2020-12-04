# detr.res50.c5.coco.multiscale.150e.bs16  

seed: 53362285

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.588
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.163
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.580
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.507
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.551
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.271
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.607
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.794
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 38.699 | 58.778 | 40.381 | 16.342 | 42.230 | 57.968 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 50.163 | bicycle      | 28.408 | car            | 35.889 |  
| motorcycle    | 41.753 | airplane     | 68.550 | bus            | 64.963 |  
| train         | 62.257 | truck        | 32.490 | boat           | 20.728 |  
| traffic light | 18.260 | fire hydrant | 62.931 | stop sign      | 56.757 |  
| parking meter | 41.841 | bench        | 23.218 | bird           | 31.951 |  
| cat           | 71.842 | dog          | 64.475 | horse          | 58.365 |  
| sheep         | 49.112 | cow          | 53.198 | elephant       | 63.878 |  
| bear          | 75.177 | zebra        | 68.942 | giraffe        | 69.895 |  
| backpack      | 11.546 | umbrella     | 36.078 | handbag        | 12.308 |  
| tie           | 30.643 | suitcase     | 33.692 | frisbee        | 57.699 |  
| skis          | 19.183 | snowboard    | 33.415 | sports ball    | 34.462 |  
| kite          | 32.113 | baseball bat | 28.803 | baseball glove | 29.507 |  
| skateboard    | 46.134 | surfboard    | 35.214 | tennis racket  | 43.606 |  
| bottle        | 28.645 | wine glass   | 28.431 | cup            | 34.601 |  
| fork          | 31.350 | knife        | 13.604 | spoon          | 16.211 |  
| bowl          | 35.670 | banana       | 23.285 | apple          | 18.351 |  
| sandwich      | 35.707 | orange       | 27.844 | broccoli       | 22.928 |  
| carrot        | 16.394 | hot dog      | 33.532 | pizza          | 49.110 |  
| donut         | 40.126 | cake         | 34.880 | chair          | 24.532 |  
| couch         | 48.334 | potted plant | 24.322 | bed            | 51.505 |  
| dining table  | 29.248 | toilet       | 61.613 | tv             | 54.620 |  
| laptop        | 59.471 | mouse        | 52.445 | remote         | 20.484 |  
| keyboard      | 49.820 | cell phone   | 26.118 | microwave      | 53.551 |  
| oven          | 36.416 | toaster      | 33.840 | sink           | 32.883 |  
| refrigerator  | 56.590 | book         | 8.930  | clock          | 42.903 |  
| vase          | 33.422 | scissors     | 36.071 | teddy bear     | 44.827 |  
| hair drier    | 6.733  | toothbrush   | 17.117 |                |        |

