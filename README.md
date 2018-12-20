# chen-models

1. What's this repository?

  It contains some unofficial codes for distributed training process of plain-CNN, CifarNet, ResNet, AlexNet, VGG and Inception-v3.
  
2. How to run the codes?

  (1) Where to place the folder?
  
    Under "/home/ubuntu".
    
  (2) Environmental Setup:
  
    export ps='ps-0-ip ps-1-ip'
    
    export workers='worker-0-ip worker-1-ip worker-2-ip'

  (3) Command to launch a distributed training process
  
    "bash train.sh <model_name> <mode_name(bsp,asp,ssp)>"

	NOTE: Some "<model>-<mode>" pairs are not yet available yet.
    
  (4) Where to find the logs?
  
    Remote worker-<i> would send the stdout log to "/home/ubuntu/worker_<i>.log".
    
    PS-<i> would send the stdout log to "/home/ubuntu/ps_<i>.log".
    
   (5) Some auxillary scripts.
   
    "bash clear.sh" to stop the distributed training process, by killing the python processes in each worker.
