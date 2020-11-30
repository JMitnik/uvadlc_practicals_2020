# Vary Learning rate
python train.py --label democracy_lrnormal --txt_file assets/book_EN_democracy_in_the_US.txt
python train.py --label democracy_lr1e-3 --learning_rate 0.002 --txt_file assets/book_EN_democracy_in_the_US.txt
python train.py --label democracy_lr2e-2 --learning_rate 0.02 --txt_file assets/book_EN_democracy_in_the_US.txt
python train.py --label democracy_lr1e-4 --learning_rate 0.0002 --txt_file assets/book_EN_democracy_in_the_US.txt

# Vary number layers
python train.py --label democracy_layers1 --lstm_num_layers 1 --txt_file assets/book_EN_democracy_in_the_US.txt
python train.py --label democracy_layers3 --lstm_num_layers 3 --txt_file assets/book_EN_democracy_in_the_US.txt

# Vary number hidden
python train.py --label democracy_numhidden64 --lstm_num_hidden 64 --txt_file assets/book_EN_democracy_in_the_US.txt
python train.py --label democracy_numhidden128 --lstm_num_hidden 128 --txt_file assets/book_EN_democracy_in_the_US.txt
python train.py --label democracy_numhidden256 --lstm_num_hidden 256 --txt_file assets/book_EN_democracy_in_the_US.txt

# Vary seq length
python train.py --label democracy_seqlength30 --seq_length 30 --txt_file assets/book_EN_democracy_in_the_US.txt
python train.py --label democracy_seqlength60 --seq_length 60 --txt_file assets/book_EN_democracy_in_the_US.txt
python train.py --label democracy_seqlength80 --seq_length 80 --txt_file assets/book_EN_democracy_in_the_US.txt
