# cascade_rcnn.res50.fpn.coco.800size.1x  

## Evaluation results for bbox:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.417
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.594
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.239
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.449
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.341
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.537
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.562
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.356
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.597
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.719
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 41.683 | 59.431 | 45.287 | 23.867 | 44.946 | 54.919 |

### Per-category bbox AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 57.027 | bicycle      | 30.835 | car            | 45.247 |  
| motorcycle    | 42.101 | airplane     | 65.692 | bus            | 65.644 |  
| train         | 65.120 | truck        | 35.008 | boat           | 27.898 |  
| traffic light | 28.982 | fire hydrant | 69.488 | stop sign      | 68.741 |  
| parking meter | 46.985 | bench        | 23.230 | bird           | 37.820 |  
| cat           | 69.089 | dog          | 62.843 | horse          | 57.619 |  
| sheep         | 50.051 | cow          | 54.645 | elephant       | 63.435 |  
| bear          | 68.021 | zebra        | 68.131 | giraffe        | 68.263 |  
| backpack      | 15.290 | umbrella     | 38.330 | handbag        | 13.166 |  
| tie           | 35.427 | suitcase     | 37.377 | frisbee        | 66.161 |  
| skis          | 23.297 | snowboard    | 37.565 | sports ball    | 48.225 |  
| kite          | 43.063 | baseball bat | 28.493 | baseball glove | 34.962 |  
| skateboard    | 50.617 | surfboard    | 38.710 | tennis racket  | 45.975 |  
| bottle        | 39.936 | wine glass   | 35.142 | cup            | 42.012 |  
| fork          | 33.924 | knife        | 18.122 | spoon          | 16.360 |  
| bowl          | 42.074 | banana       | 24.342 | apple          | 19.396 |  
| sandwich      | 35.107 | orange       | 32.242 | broccoli       | 23.533 |  
| carrot        | 22.634 | hot dog      | 33.035 | pizza          | 53.193 |  
| donut         | 44.881 | cake         | 35.677 | chair          | 26.735 |  
| couch         | 42.121 | potted plant | 25.568 | bed            | 43.200 |  
| dining table  | 27.826 | toilet       | 60.139 | tv             | 55.109 |  
| laptop        | 60.335 | mouse        | 63.750 | remote         | 29.355 |  
| keyboard      | 52.907 | cell phone   | 34.924 | microwave      | 56.181 |  
| oven          | 32.588 | toaster      | 42.032 | sink           | 37.071 |  
| refrigerator  | 58.243 | book         | 16.024 | clock          | 52.030 |  
| vase          | 37.956 | scissors     | 28.882 | teddy bear     | 47.347 |  
| hair drier    | 2.970  | toothbrush   | 21.168 |                |        |


## Evaluation results for segm:  

```  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.361
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.565
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.384
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.466
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.290
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.642
```  
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |  
|:------:|:------:|:------:|:------:|:------:|:------:|  
| 36.104 | 56.472 | 38.855 | 17.604 | 38.426 | 52.229 |

### Per-category segm AP:  

| category      | AP     | category     | AP     | category       | AP     |  
|:--------------|:-------|:-------------|:-------|:---------------|:-------|  
| person        | 46.314 | bicycle      | 16.598 | car            | 39.891 |  
| motorcycle    | 30.739 | airplane     | 48.127 | bus            | 61.753 |  
| train         | 60.895 | truck        | 32.687 | boat           | 22.617 |  
| traffic light | 26.710 | fire hydrant | 63.170 | stop sign      | 66.614 |  
| parking meter | 46.772 | bench        | 16.393 | bird           | 28.961 |  
| cat           | 66.090 | dog          | 57.746 | horse          | 39.252 |  
| sheep         | 41.812 | cow          | 45.328 | elephant       | 55.834 |  
| bear          | 65.349 | zebra        | 56.590 | giraffe        | 49.582 |  
| backpack      | 13.530 | umbrella     | 43.347 | handbag        | 12.916 |  
| tie           | 30.581 | suitcase     | 37.657 | frisbee        | 62.812 |  
| skis          | 2.386  | snowboard    | 21.604 | sports ball    | 46.260 |  
| kite          | 29.643 | baseball bat | 20.983 | baseball glove | 36.700 |  
| skateboard    | 29.038 | surfboard    | 30.214 | tennis racket  | 51.730 |  
| bottle        | 37.192 | wine glass   | 30.028 | cup            | 41.150 |  
| fork          | 13.669 | knife        | 10.499 | spoon          | 9.554  |  
| bowl          | 37.978 | banana       | 18.596 | apple          | 17.717 |  
| sandwich      | 35.520 | orange       | 30.800 | broccoli       | 21.839 |  
| carrot        | 19.070 | hot dog      | 25.079 | pizza          | 50.794 |  
| donut         | 44.040 | cake         | 34.434 | chair          | 16.861 |  
| couch         | 34.763 | potted plant | 21.782 | bed            | 32.989 |  
| dining table  | 15.064 | toilet       | 57.908 | tv             | 55.626 |  
| laptop        | 57.852 | mouse        | 61.913 | remote         | 26.012 |  
| keyboard      | 49.774 | cell phone   | 32.369 | microwave      | 55.606 |  
| oven          | 28.981 | toaster      | 46.036 | sink           | 33.469 |  
| refrigerator  | 57.836 | book         | 10.192 | clock          | 51.035 |  
| vase          | 35.635 | scissors     | 19.337 | teddy bear     | 42.266 |  
| hair drier    | 0.000  | toothbrush   | 11.801 |                |        |
