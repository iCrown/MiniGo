WHITE="gnugo --mode gtp"
BLACK="python main.py gtp policy --read-file=checkpoint/v3_clean_l7/epoch_48.ckpt"
gogui-twogtp -black "$BLACK" -white "$WHITE" -games 10 -size 9 -sgffile gnugo -auto
