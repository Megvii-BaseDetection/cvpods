# pointrend.res50.fpn.coco.multiscale.1x  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.384
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.591
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.339
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.565
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.667
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 38.389 | 59.093 | 41.982 | 22.438 | 41.579 | 49.848 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 53.659 | bicycle      | 28.958 | car            | 42.951 |  
| motorcycle    | 39.469 | airplane     | 59.767 | bus            | 62.232 |  
| train         | 58.041 | truck        | 31.652 | boat           | 26.171 |  
| traffic light | 27.019 | fire hydrant | 66.369 | stop sign      | 62.603 |  
| parking meter | 46.385 | bench        | 21.754 | bird           | 35.167 |  
| cat           | 62.504 | dog          | 57.641 | horse          | 53.234 |  
| sheep         | 47.952 | cow          | 52.071 | elephant       | 58.470 |  
| bear          | 67.019 | zebra        | 64.606 | giraffe        | 64.318 |  
| backpack      | 13.696 | umbrella     | 35.344 | handbag        | 11.628 |  
| tie           | 30.786 | suitcase     | 33.192 | frisbee        | 60.511 |  
| skis          | 19.795 | snowboard    | 32.499 | sports ball    | 46.719 |  
| kite          | 40.466 | baseball bat | 24.203 | baseball glove | 33.456 |  
| skateboard    | 44.681 | surfboard    | 35.043 | tennis racket  | 43.413 |  
| bottle        | 38.526 | wine glass   | 33.946 | cup            | 39.267 |  
| fork          | 28.034 | knife        | 15.859 | spoon          | 12.625 |  
| bowl          | 41.594 | banana       | 22.949 | apple          | 18.706 |  
| sandwich      | 32.218 | orange       | 28.354 | broccoli       | 22.721 |  
| carrot        | 20.691 | hot dog      | 25.885 | pizza          | 50.169 |  
| donut         | 40.519 | cake         | 33.393 | chair          | 24.791 |  
| couch         | 36.772 | potted plant | 24.788 | bed            | 37.107 |  
| dining table  | 25.053 | toilet       | 54.735 | tv             | 52.190 |  
| laptop        | 55.395 | mouse        | 60.509 | remote         | 26.056 |  
| keyboard      | 48.161 | cell phone   | 33.062 | microwave      | 50.259 |  
| oven          | 30.547 | toaster      | 37.250 | sink           | 32.427 |  
| refrigerator  | 51.783 | book         | 13.965 | clock          | 48.457 |  
| vase          | 36.217 | scissors     | 24.865 | teddy bear     | 43.867 |  
| hair drier    | 5.302  | toothbrush   | 16.658 |                |        |


## Evaluation results for segm:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.566
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.308
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.532
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.660
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 36.236 | 56.632 | 38.616 | 17.319 | 38.595 | 52.869 |

### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 47.754 | bicycle      | 17.643 | car            | 40.412 |  
| motorcycle    | 32.427 | airplane     | 50.226 | bus            | 63.716 |  
| train         | 60.307 | truck        | 32.388 | boat           | 22.870 |  
| traffic light | 26.512 | fire hydrant | 64.840 | stop sign      | 63.625 |  
| parking meter | 48.243 | bench        | 17.240 | bird           | 29.486 |  
| cat           | 66.955 | dog          | 57.118 | horse          | 40.660 |  
| sheep         | 42.643 | cow          | 46.251 | elephant       | 56.319 |  
| bear          | 67.436 | zebra        | 57.974 | giraffe        | 53.530 |  
| backpack      | 14.966 | umbrella     | 43.557 | handbag        | 12.993 |  
| tie           | 31.080 | suitcase     | 37.062 | frisbee        | 60.350 |  
| skis          | 3.277  | snowboard    | 23.533 | sports ball    | 46.489 |  
| kite          | 30.256 | baseball bat | 22.547 | baseball glove | 37.711 |  
| skateboard    | 31.399 | surfboard    | 30.683 | tennis racket  | 53.293 |  
| bottle        | 37.495 | wine glass   | 30.994 | cup            | 40.058 |  
| fork          | 14.071 | knife        | 10.168 | spoon          | 9.365  |  
| bowl          | 39.559 | banana       | 19.776 | apple          | 18.915 |  
| sandwich      | 34.329 | orange       | 28.827 | broccoli       | 22.166 |  
| carrot        | 18.897 | hot dog      | 19.813 | pizza          | 49.496 |  
| donut         | 42.555 | cake         | 35.128 | chair          | 17.089 |  
| couch         | 32.357 | potted plant | 21.316 | bed            | 29.910 |  
| dining table  | 13.514 | toilet       | 57.944 | tv             | 55.088 |  
| laptop        | 58.292 | mouse        | 61.422 | remote         | 26.752 |  
| keyboard      | 49.727 | cell phone   | 32.720 | microwave      | 53.465 |  
| oven          | 31.089 | toaster      | 42.077 | sink           | 32.144 |  
| refrigerator  | 54.004 | book         | 9.596  | clock          | 49.843 |  
| vase          | 35.966 | scissors     | 21.042 | teddy bear     | 43.754 |  
| hair drier    | 0.617  | toothbrush   | 11.768 |                |        |
