# Vary Learning rate
python train.py --label grimm_potential_pls_lrnormal --batch_size 256 --learning_rate 0.02 --lstm_num_hidden 512 --txt_file assets/book_EN_grimms_fairy_tails.txt
# python train.py --label democracy_lr1e-3 --learning_rate 0.002 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label democracy_lr2e-2 --learning_rate 0.02 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label democracy_lr1e-4 --learning_rate 0.0002 --txt_file assets/book_EN_democracy_in_the_US.txt

# # Vary number layers
# python train.py --label democracy_layers1 --lstm_num_layers 1 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label democracy_layers3 --lstm_num_layers 3 --txt_file assets/book_EN_democracy_in_the_US.txt

# # Vary number hidden
# python train.py --label democracy_numhidden64 --lstm_num_hidden 64 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label democracy_numhidden128 --lstm_num_hidden 128 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label democracy_numhidden256 --lstm_num_hidden 256 --txt_file assets/book_EN_democracy_in_the_US.txt

# # Vary seq length
# python train.py --label democracy_seqlength30 --seq_length 30 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label democracy_seqlength60 --seq_length 60 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label democracy_seqlength80 --seq_length 80 --txt_file assets/book_EN_democracy_in_the_US.txt

# # Vary sentence length generation
# python train.py --label democracy_gen30 --nr_to_sample 30 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label democracy_gen50 --nr_to_sample 50 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label democracy_gen70 --nr_to_sample 70 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label darwin_reis_gen30 --nr_to_sample 30 --txt_file assets/book_NL_darwin_reis_om_de_wereld.txt
# python train.py --label darwin_reis_gen50 --nr_to_sample 50 --txt_file assets/book_NL_darwin_reis_om_de_wereld.txt
# python train.py --label darwin_reis_gen70 --nr_to_sample 70 --txt_file assets/book_NL_darwin_reis_om_de_wereld.txt

# # Vary temperature
# python train.py --label democracy_gentemp1 --temperature 1 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label democracy_gentemp0.5 --temperature 0.5 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label democracy_gentemp2 --temperature 2 --txt_file assets/book_EN_democracy_in_the_US.txt
# python train.py --label darwin_reis_gentemp1 --temperature 1 --txt_file assets/book_NL_darwin_reis_om_de_wereld.txt
# python train.py --label darwin_reis_gentemp0.5 --temperature 0.5 --txt_file assets/book_NL_darwin_reis_om_de_wereld.txt
# python train.py --label darwin_reis_gentemp2 --temperature 2 --txt_file assets/book_NL_darwin_reis_om_de_wereld.txt
