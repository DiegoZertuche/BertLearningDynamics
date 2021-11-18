cd pre-trained_language_models/

for seed in 0 1 2 3 4
  do
  for ckpt in 0 20 60 100 140 200 400 700 1000 1500 1800 2000
    do
    echo "MultiBERTs Seed ${seed} Step ${ckpt}k"
    if [[ ! -f bert/multiberts-seed_${seed}-step_${ckpt}k/bert_config.json ]]; then
      mkdir -p 'bert'
      cd bert
      git clone https://huggingface.co/google/multiberts-seed_${seed}-step_${ckpt}k
      chmod +x multiberts-seed_${seed}-step_${ckpt}k/.git/hooks/post-checkout
      cd multiberts-seed_${seed}-step_${ckpt}k
      mv config.json bert_config.json
      cd ../../
    fi
    done
  done


