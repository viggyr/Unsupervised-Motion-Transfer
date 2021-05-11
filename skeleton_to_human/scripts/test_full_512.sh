############## To test full model #############
##### Using GPUs with 12G memory (not tested)
export CUDA_VISIBLE_DEVICES=0
/opt/conda/bin/python test_video.py --name everybody_dance_now_temporal --model pose2vid --dataroot ./datasets/test_target/ --which_epoch latest --netG local --ngf 32 --label_nc 0 --no_instance --resize_or_crop none
