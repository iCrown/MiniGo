BLACK="gnugo --mode gtp --level 1"
WHITE="python main.py gtp policy model/rl/52/player52.ckpt PlayerNetwork"
TWOGTP="../gogui-1.4.9/bin/gogui-twogtp -black \"$BLACK\" -white \"$WHITE\" -games 100 -size 9 -alternate -sgffile gnugo -force"
../gogui-1.4.9/bin/gogui -size 9 -program "$TWOGTP" -computer-both -auto
