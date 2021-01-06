## Evaluation results for bbox:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.583
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.314
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.492
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.312
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.552
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.662
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 37.431 | 58.324 | 40.753 | 20.488 | 40.542 | 49.500 |
### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 52.348 | bicycle      | 27.992 | car            | 41.449 |  
| motorcycle    | 37.357 | airplane     | 56.897 | bus            | 60.740 |  
| train         | 59.973 | truck        | 30.468 | boat           | 25.493 |  
| traffic light | 26.055 | fire hydrant | 63.453 | stop sign      | 63.956 |  
| parking meter | 42.996 | bench        | 21.513 | bird           | 32.954 |  
| cat           | 58.961 | dog          | 54.468 | horse          | 53.189 |  
| sheep         | 46.781 | cow          | 50.254 | elephant       | 58.252 |  
| bear          | 63.393 | zebra        | 62.759 | giraffe        | 61.757 |  
| backpack      | 12.170 | umbrella     | 34.308 | handbag        | 10.634 |  
| tie           | 29.113 | suitcase     | 32.423 | frisbee        | 59.787 |  
| skis          | 18.986 | snowboard    | 29.724 | sports ball    | 46.825 |  
| kite          | 39.245 | baseball bat | 22.023 | baseball glove | 33.063 |  
| skateboard    | 43.847 | surfboard    | 33.287 | tennis racket  | 42.123 |  
| bottle        | 36.359 | wine glass   | 32.813 | cup            | 38.227 |  
| fork          | 26.179 | knife        | 13.666 | spoon          | 13.162 |  
| bowl          | 40.316 | banana       | 23.297 | apple          | 18.609 |  
| sandwich      | 29.270 | orange       | 27.714 | broccoli       | 21.913 |  
| carrot        | 20.149 | hot dog      | 27.557 | pizza          | 48.518 |  
| donut         | 40.494 | cake         | 33.266 | chair          | 24.138 |  
| couch         | 37.527 | potted plant | 24.376 | bed            | 40.752 |  
| dining table  | 26.393 | toilet       | 54.339 | tv             | 52.574 |  
| laptop        | 55.793 | mouse        | 58.448 | remote         | 23.667 |  
| keyboard      | 47.191 | cell phone   | 32.427 | microwave      | 53.062 |  
| oven          | 29.422 | toaster      | 35.287 | sink           | 33.171 |  
| refrigerator  | 51.050 | book         | 14.176 | clock          | 47.204 |  
| vase          | 35.532 | scissors     | 21.997 | teddy bear     | 43.649 |  
| hair drier    | 3.366  | toothbrush   | 16.433 |                |        |
## Evaluation results for segm:  
```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.554
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.369
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.295
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.473
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.508
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.626
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 34.331 | 55.357 | 36.552 | 15.369 | 36.896 | 50.237 |
### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 44.892 | bicycle      | 16.207 | car            | 38.319 |  
| motorcycle    | 28.910 | airplane     | 46.307 | bus            | 59.838 |  
| train         | 59.586 | truck        | 31.387 | boat           | 21.929 |  
| traffic light | 25.158 | fire hydrant | 60.934 | stop sign      | 64.527 |  
| parking meter | 45.093 | bench        | 15.568 | bird           | 27.505 |  
| cat           | 62.278 | dog          | 54.367 | horse          | 38.481 |  
| sheep         | 41.140 | cow          | 43.623 | elephant       | 54.351 |  
| bear          | 62.252 | zebra        | 54.134 | giraffe        | 46.853 |  
| backpack      | 12.601 | umbrella     | 42.482 | handbag        | 11.692 |  
| tie           | 28.097 | suitcase     | 33.311 | frisbee        | 60.049 |  
| skis          | 1.932  | snowboard    | 19.151 | sports ball    | 45.731 |  
| kite          | 29.837 | baseball bat | 19.292 | baseball glove | 36.700 |  
| skateboard    | 26.734 | surfboard    | 26.909 | tennis racket  | 50.354 |  
| bottle        | 35.078 | wine glass   | 28.633 | cup            | 38.918 |  
| fork          | 10.819 | knife        | 8.586  | spoon          | 9.127  |  
| bowl          | 38.163 | banana       | 18.516 | apple          | 17.697 |  
| sandwich      | 31.305 | orange       | 28.214 | broccoli       | 20.902 |  
| carrot        | 17.833 | hot dog      | 20.497 | pizza          | 47.327 |  
| donut         | 40.921 | cake         | 33.916 | chair          | 15.530 |  
| couch         | 32.646 | potted plant | 21.622 | bed            | 32.449 |  
| dining table  | 14.536 | toilet       | 56.432 | tv             | 55.255 |  
| laptop        | 56.120 | mouse        | 58.399 | remote         | 23.077 |  
| keyboard      | 47.962 | cell phone   | 32.144 | microwave      | 52.815 |  
| oven          | 27.298 | toaster      | 37.841 | sink           | 31.680 |  
| refrigerator  | 53.089 | book         | 8.918  | clock          | 48.439 |  
| vase          | 34.445 | scissors     | 15.931 | teddy bear     | 41.777 |  
| hair drier    | 0.361  | toothbrush   | 12.779 |                |        |
Panoptic Evaluation Results:
```  
|        |   PQ   |   SQ   |   RQ   |  #categories  |  
|:------:|:------:|:------:|:------:|:-------------:|  
|  All   | 39.464 | 78.075 | 48.343 |      133      |  
| Things | 45.919 | 80.979 | 55.285 |      80       |  
| Stuff  | 29.721 | 73.692 | 37.864 |      53       |