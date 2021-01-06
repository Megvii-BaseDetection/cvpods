# centernet.res50.coco.512size.new  

seed: 31799395

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.348
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.537
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.526
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.469
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.235
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.724
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 34.797 | 53.668 | 36.952 | 14.451 | 39.879 | 52.622 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 46.551 | bicycle      | 23.358 | car            | 33.902 |  
| motorcycle    | 36.949 | airplane     | 57.914 | bus            | 58.833 |  
| train         | 58.629 | truck        | 29.757 | boat           | 20.882 |  
| traffic light | 20.124 | fire hydrant | 56.920 | stop sign      | 57.538 |  
| parking meter | 41.997 | bench        | 19.343 | bird           | 27.422 |  
| cat           | 62.343 | dog          | 57.934 | horse          | 55.077 |  
| sheep         | 46.631 | cow          | 49.524 | elephant       | 59.045 |  
| bear          | 68.447 | zebra        | 64.142 | giraffe        | 64.267 |  
| backpack      | 11.443 | umbrella     | 34.201 | handbag        | 10.398 |  
| tie           | 23.916 | suitcase     | 33.467 | frisbee        | 47.884 |  
| skis          | 17.515 | snowboard    | 24.925 | sports ball    | 32.255 |  
| kite          | 34.981 | baseball bat | 21.676 | baseball glove | 28.879 |  
| skateboard    | 45.539 | surfboard    | 30.550 | tennis racket  | 41.678 |  
| bottle        | 29.652 | wine glass   | 26.433 | cup            | 33.370 |  
| fork          | 26.267 | knife        | 11.744 | spoon          | 11.408 |  
| bowl          | 32.179 | banana       | 19.605 | apple          | 15.493 |  
| sandwich      | 30.437 | orange       | 26.654 | broccoli       | 16.370 |  
| carrot        | 16.292 | hot dog      | 28.934 | pizza          | 44.213 |  
| donut         | 39.721 | cake         | 32.999 | chair          | 23.426 |  
| couch         | 39.998 | potted plant | 23.893 | bed            | 34.584 |  
| dining table  | 18.389 | toilet       | 58.024 | tv             | 50.795 |  
| laptop        | 54.527 | mouse        | 49.496 | remote         | 17.014 |  
| keyboard      | 45.236 | cell phone   | 23.726 | microwave      | 53.201 |  
| oven          | 29.616 | toaster      | 15.225 | sink           | 31.995 |  
| refrigerator  | 51.812 | book         | 6.289  | clock          | 42.856 |  
| vase          | 30.781 | scissors     | 21.761 | teddy bear     | 40.987 |  
| hair drier    | 6.018  | toothbrush   | 15.477 |                |        |
