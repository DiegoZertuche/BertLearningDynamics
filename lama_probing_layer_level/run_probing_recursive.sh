for seed in 0 1 2 3 4
  do
  for ckpt in 0 20 60 100 140 400 700 1500 2000
    do
    for layer in 1 2 4 6 8 10 12
    do
    echo "MultiBERTs Seed ${seed} Step ${ckpt}k Layer ${layer}"
    python run_probing.py \
            --decoder_type Huggingface_pretrained_decoder \
            --do_probing \
            --probing_layer ${layer} \
            --use_model_from_dir \
            --model_dir /content/drive/MyDrive/G2\ 2021\ Fall/AC\ 297R\ Capstone/BERTnesia-knowledge-probing-master/pre-trained_language_models/bert/multiberts-seed_${seed}-step_${ckpt}k
    done
  done
done