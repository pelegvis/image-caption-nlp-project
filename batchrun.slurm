#! /bin/bash

#SBATCH --job-name=IMGCAP
#SBATCH --output=IMGCAP.out
#SBATCH --error=IMGCAP.err
#SBATCH --partition=studentrun
#SBATCH --time=170
#SBATCH --gpus=1
source /home/yandex/DLW2021/davidhay/anaconda3/bin/activate
python main.py --batch 3 --progress 5000 --checkpoint "/home/yandex/DLW2021/davidhay/checkpoint.pt" "/home/yandex/DLW2021/davidhay/coco/train2017" "/home/yandex/DLW2021/davidhay/coco/annotations/captions_train2017.json" 40
