# tensormask.res50.fpn.coco.800size.1x  
## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.375
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.564
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.204
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.403
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.493
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.323
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.524
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.357
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.587
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.711
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 37.484 | 56.399 | 40.494 | 20.430 | 40.310 | 49.281 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 53.485 | bicycle      | 28.358 | car            | 43.793 |  
| motorcycle    | 39.019 | airplane     | 60.730 | bus            | 62.679 |  
| train         | 61.183 | truck        | 31.826 | boat           | 25.288 |  
| traffic light | 27.092 | fire hydrant | 64.775 | stop sign      | 65.273 |  
| parking meter | 38.983 | bench        | 20.623 | bird           | 35.048 |  
| cat           | 63.591 | dog          | 56.913 | horse          | 50.671 |  
| sheep         | 47.719 | cow          | 51.395 | elephant       | 59.859 |  
| bear          | 67.612 | zebra        | 66.866 | giraffe        | 65.176 |  
| backpack      | 13.246 | umbrella     | 35.374 | handbag        | 11.343 |  
| tie           | 30.673 | suitcase     | 32.243 | frisbee        | 62.134 |  
| skis          | 19.634 | snowboard    | 24.306 | sports ball    | 46.279 |  
| kite          | 41.484 | baseball bat | 20.534 | baseball glove | 32.225 |  
| skateboard    | 45.949 | surfboard    | 30.247 | tennis racket  | 42.709 |  
| bottle        | 34.758 | wine glass   | 32.058 | cup            | 38.543 |  
| fork          | 24.448 | knife        | 13.198 | spoon          | 8.293  |  
| bowl          | 37.472 | banana       | 20.896 | apple          | 19.170 |  
| sandwich      | 30.304 | orange       | 28.257 | broccoli       | 22.745 |  
| carrot        | 18.222 | hot dog      | 27.828 | pizza          | 46.730 |  
| donut         | 41.520 | cake         | 30.753 | chair          | 23.458 |  
| couch         | 38.522 | potted plant | 23.999 | bed            | 41.425 |  
| dining table  | 24.882 | toilet       | 59.175 | tv             | 53.753 |  
| laptop        | 53.677 | mouse        | 61.380 | remote         | 26.240 |  
| keyboard      | 45.266 | cell phone   | 32.449 | microwave      | 55.397 |  
| oven          | 32.160 | toaster      | 14.698 | sink           | 32.348 |  
| refrigerator  | 48.487 | book         | 11.438 | clock          | 51.231 |  
| vase          | 34.637 | scissors     | 22.842 | teddy bear     | 44.466 |  
| hair drier    | 0.764  | toothbrush   | 14.479 |                |        |
## Evaluation results for segm:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.323
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.524
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.142
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.344
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.287
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.478
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.284
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.512
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.637
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 32.255 | 52.357 | 33.909 | 14.179 | 34.356 | 46.384 |
### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 42.666 | bicycle      | 14.352 | car            | 38.916 |  
| motorcycle    | 26.537 | airplane     | 46.410 | bus            | 60.674 |  
| train         | 59.607 | truck        | 31.132 | boat           | 19.813 |  
| traffic light | 24.946 | fire hydrant | 60.417 | stop sign      | 63.551 |  
| parking meter | 41.287 | bench        | 13.001 | bird           | 26.686 |  
| cat           | 58.934 | dog          | 53.548 | horse          | 37.104 |  
| sheep         | 39.993 | cow          | 42.220 | elephant       | 53.463 |  
| bear          | 64.799 | zebra        | 57.484 | giraffe        | 51.431 |  
| backpack      | 13.066 | umbrella     | 40.720 | handbag        | 9.757  |  
| tie           | 25.684 | suitcase     | 29.698 | frisbee        | 57.956 |  
| skis          | 2.362  | snowboard    | 14.886 | sports ball    | 42.948 |  
| kite          | 25.752 | baseball bat | 14.991 | baseball glove | 34.131 |  
| skateboard    | 24.034 | surfboard    | 19.352 | tennis racket  | 45.952 |  
| bottle        | 31.177 | wine glass   | 26.017 | cup            | 37.558 |  
| fork          | 8.464  | knife        | 7.151  | spoon          | 3.512  |  
| bowl          | 33.670 | banana       | 16.717 | apple          | 17.550 |  
| sandwich      | 31.199 | orange       | 26.635 | broccoli       | 20.435 |  
| carrot        | 14.255 | hot dog      | 20.054 | pizza          | 43.201 |  
| donut         | 40.383 | cake         | 29.727 | chair          | 14.863 |  
| couch         | 30.694 | potted plant | 19.266 | bed            | 30.556 |  
| dining table  | 12.695 | toilet       | 57.141 | tv             | 55.168 |  
| laptop        | 52.058 | mouse        | 59.619 | remote         | 22.521 |  
| keyboard      | 40.731 | cell phone   | 28.858 | microwave      | 53.165 |  
| oven          | 27.610 | toaster      | 17.222 | sink           | 28.681 |  
| refrigerator  | 46.344 | book         | 4.869  | clock          | 49.094 |  
| vase          | 32.431 | scissors     | 10.445 | teddy bear     | 42.369 |  
| hair drier    | 0.464  | toothbrush   | 5.578  |                |        |
