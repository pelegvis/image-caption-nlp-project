#! /bin/bash

#SBATCH --job-name=DLWSHP
#SBATCH --output=DLWSHP.out
#SBATCH --error=DLWSHP.err
#SBATCH --partition=studentrun
#SBATCH --time=180
#SBATCH --gpus=1
source /home/yandex/DLW2021/pelegv/anaconda3/bin/activate
# python main.py --batch 3 --progress 5000 --load_vocab --checkpoint "/home/yandex/DLW2021/pelegv/checkpoint.pt" "/home/yandex/DLW2021/pelegv/coco/train2017" "/home/yandex/DLW2021/pelegv/coco/annotations/captions_train2017.json" 20
python main.py --load_vocab --overfit "/home/yandex/DLW2021/pelegv/coco/train2017" "/home/yandex/DLW2021/pelegv/coco/annotations/captions_train2017.json" 1
