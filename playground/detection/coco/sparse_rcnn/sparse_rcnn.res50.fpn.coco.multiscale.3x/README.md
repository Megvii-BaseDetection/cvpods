# sparse_rcnn.res50.fpn.coco.multiscale.3x

seed: 40244023

## Evaluation results for bbox:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.432
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.620
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.466
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.586
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.350
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.607
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.794
```
|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 43.170 | 61.980 | 46.637 | 25.439 | 45.547 | 58.599 |

### Per-category bbox AP:

| category      | AP     | category     | AP     | category       | AP     |
|:--------------|:-------|:-------------|:-------|:---------------|:-------|
| person        | 54.616 | bicycle      | 28.852 | car            | 44.614 |
| motorcycle    | 44.515 | airplane     | 70.130 | bus            | 65.510 |
| train         | 62.451 | truck        | 35.686 | boat           | 27.032 |
| traffic light | 27.065 | fire hydrant | 63.710 | stop sign      | 66.756 |
| parking meter | 51.703 | bench        | 27.281 | bird           | 37.681 |
| cat           | 72.674 | dog          | 66.336 | horse          | 61.524 |
| sheep         | 53.165 | cow          | 60.175 | elephant       | 68.559 |
| bear          | 74.711 | zebra        | 69.616 | giraffe        | 67.261 |
| backpack      | 16.017 | umbrella     | 39.189 | handbag        | 16.653 |
| tie           | 32.789 | suitcase     | 42.350 | frisbee        | 68.649 |
| skis          | 24.925 | snowboard    | 41.256 | sports ball    | 48.665 |
| kite          | 43.454 | baseball bat | 31.019 | baseball glove | 38.397 |
| skateboard    | 55.856 | surfboard    | 40.121 | tennis racket  | 49.532 |
| bottle        | 37.018 | wine glass   | 35.598 | cup            | 42.640 |
| fork          | 37.525 | knife        | 20.016 | spoon          | 18.674 |
| bowl          | 42.556 | banana       | 24.694 | apple          | 18.817 |
| sandwich      | 37.074 | orange       | 31.126 | broccoli       | 22.715 |
| carrot        | 17.282 | hot dog      | 35.849 | pizza          | 54.289 |
| donut         | 46.511 | cake         | 35.549 | chair          | 26.069 |
| couch         | 46.136 | potted plant | 24.983 | bed            | 47.052 |
| dining table  | 30.304 | toilet       | 63.607 | tv             | 57.375 |
| laptop        | 58.124 | mouse        | 63.222 | remote         | 32.923 |
| keyboard      | 52.616 | cell phone   | 37.213 | microwave      | 62.165 |
| oven          | 36.953 | toaster      | 34.455 | sink           | 36.135 |
| refrigerator  | 57.358 | book         | 11.503 | clock          | 51.863 |
| vase          | 40.804 | scissors     | 38.068 | teddy bear     | 46.949 |
| hair drier    | 17.578 | toothbrush   | 31.711 |                |        |