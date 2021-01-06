# ssd.vgg16.coco.300size.expand_aug  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.249
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.259
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.062
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.274
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.079
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.404
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.588
```  
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |  
|:------:|:------:|:------:|:-----:|:------:|:------:|  
| 24.878 | 41.571 | 25.903 | 6.154 | 27.351 | 43.304 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 32.729 | bicycle      | 17.399 | car            | 19.948 |  
| motorcycle    | 28.864 | airplane     | 48.108 | bus            | 51.582 |  
| train         | 55.852 | truck        | 25.023 | boat           | 11.470 |  
| traffic light | 6.214  | fire hydrant | 44.395 | stop sign      | 47.489 |  
| parking meter | 33.224 | bench        | 15.225 | bird           | 18.530 |  
| cat           | 54.324 | dog          | 50.141 | horse          | 41.554 |  
| sheep         | 31.408 | cow          | 31.712 | elephant       | 48.899 |  
| bear          | 57.134 | zebra        | 50.264 | giraffe        | 51.875 |  
| backpack      | 4.930  | umbrella     | 23.523 | handbag        | 3.994  |  
| tie           | 12.083 | suitcase     | 17.553 | frisbee        | 29.053 |  
| skis          | 8.962  | snowboard    | 12.263 | sports ball    | 10.062 |  
| kite          | 15.983 | baseball bat | 11.747 | baseball glove | 12.568 |  
| skateboard    | 27.503 | surfboard    | 17.281 | tennis racket  | 25.960 |  
| bottle        | 12.138 | wine glass   | 13.746 | cup            | 20.045 |  
| fork          | 11.557 | knife        | 3.839  | spoon          | 3.355  |  
| bowl          | 26.285 | banana       | 13.517 | apple          | 9.321  |  
| sandwich      | 29.394 | orange       | 19.842 | broccoli       | 15.144 |  
| carrot        | 9.981  | hot dog      | 21.651 | pizza          | 38.271 |  
| donut         | 26.320 | cake         | 21.064 | chair          | 13.403 |  
| couch         | 35.282 | potted plant | 13.499 | bed            | 39.236 |  
| dining table  | 22.534 | toilet       | 48.075 | tv             | 44.400 |  
| laptop        | 44.153 | mouse        | 31.977 | remote         | 7.648  |  
| keyboard      | 35.873 | cell phone   | 19.075 | microwave      | 36.640 |  
| oven          | 28.469 | toaster      | 0.764  | sink           | 21.440 |  
| refrigerator  | 35.989 | book         | 3.813  | clock          | 29.932 |  
| vase          | 18.325 | scissors     | 19.013 | teddy bear     | 32.997 |  
| hair drier    | 0.000  | toothbrush   | 5.431  |                |        |
