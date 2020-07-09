python keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights "./_pretrained_model.h5" --batch-size 8 --steps 500 --epochs 10 csv annotations.csv classes.csv

python keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights "./_pretrained_model.h5" --batch-size 2 --steps 500 --epochs 10 csv annotations.csv classes.csv
