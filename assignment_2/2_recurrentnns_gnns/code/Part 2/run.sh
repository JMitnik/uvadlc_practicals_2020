python train.py --label generate-tolstoy-tnorm-30long --temperature 1.0 --txt_file assets/book_NL_tolstoy_anna_karenina.txt
python train.py --label generate-tolstoy-tnorm-40long --nr_to_sample 40 --temperature 1.0 --txt_file assets/book_NL_tolstoy_anna_karenina.txt
python train.py --label generate-tolstoy-tnorm-50long --nr_to_sample 50 --temperature 1.0 --txt_file assets/book_NL_tolstoy_anna_karenina.txt

python train.py --label generate-CSS-tnorm-30long --temperature 1.0 --txt_file assets/test.css

python train.py --label generate-tolstoy-tlow_30long --temperature 0.5 --txt_file assets/book_NL_tolstoy_anna_karenina.txt
python train.py --label generate-tolstoy-thigh-30long --temperature 2.0 --txt_file assets/book_NL_tolstoy_anna_karenina.txt

