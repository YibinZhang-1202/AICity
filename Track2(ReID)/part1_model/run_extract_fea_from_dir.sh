CUDA_VISIBLE_DEVICES=1 \
python reid/extract_fea_from_dir.py \
    --batch_size 16 \
    --resume output/cross_entropy_trihard_resnet101/pass100.pth.tar \
    --arch cross_entropy_trihard_resnet101

