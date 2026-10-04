export CUDA_VISIBLE_DEVICES=0

case="bicycle"
OUTPUT_DIR="mipnerf360"

python train.py \
    -s dataset/3dgs/mipnerf360/$case \
    -m output/3dgs/$OUTPUT_DIR/$case \
    --images images_4 \
    --iterations 30000

feature_levels="1 2 3"

for i in $feature_levels
do
    echo "Processing feature level $i"

    python gaussian_feature_extractor.py \
        -m output/3dgs/$OUTPUT_DIR/$case \
        --iteration 30000 \
        --feature_level $i \
        --use_efficient \
        --eval \
        --images images_4
done
