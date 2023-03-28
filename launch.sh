ROOT=$(dirname $(realpath ${0}))

source ${ROOT}/anaconda/bin/activate ${ROOT}/anaconda

if [ ${1} = fashioniq ]; then
  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/dress_val.json data/shirt_val.json data/toptee_val.json \
    --evaluation_retrieval_scheme 1 0 \
    --evaluation_metric_ks 10 50 \
    --optimization_split_paths data/dress_train.json data/shirt_train.json data/toptee_train.json \
    --backbone_name ${2} \
    --code_size 512 \
    --overlap_loss_weight 0.3 \
    --initial_temperature 1.0 \
    --use_gated_fusion True \
    --use_heuristic_negative True \
    --backbone_activity 1e-3 \
    --learning_rate 1e-4 \
    --warmup_epoch_count 6 \
    --total_epoch_count 64 \
    --batch_size 64 \
    --tcm_loss_weight 0.2 \
    --itm_loss_weight 0.1

  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/dress_val.json data/shirt_val.json data/toptee_val.json \
    --evaluation_retrieval_scheme 1 0 \
    --evaluation_metric_ks 10 50

  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/dress_val.json \
    --evaluation_retrieval_scheme 1 0 \
    --evaluation_metric_ks 10 50

  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/shirt_val.json \
    --evaluation_retrieval_scheme 1 0 \
    --evaluation_metric_ks 10 50

  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/toptee_val.json \
    --evaluation_retrieval_scheme 1 0 \
    --evaluation_metric_ks 10 50

elif [ ${1} = fashion200k ]; then
  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/all_test.json \
    --evaluation_retrieval_scheme 1 0 \
    --evaluation_metric_ks 1 10 50 \
    --optimization_split_paths data/all_train.json \
    --backbone_name ${2} \
    --code_size 512 \
    --overlap_loss_weight 0.3 \
    --initial_temperature 1.0 \
    --use_gated_fusion True \
    --use_heuristic_negative True \
    --backbone_activity 1e-3 \
    --learning_rate 1e-4 \
    --warmup_epoch_count 6 \
    --total_epoch_count 64 \
    --batch_size 64 \
    --tcm_loss_weight 0.2 \
    --itm_loss_weight 0.1

  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/all_test.json \
    --evaluation_retrieval_scheme 1 0 \
    --evaluation_metric_ks 1 10 50

  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/all_test.json \
    --evaluation_retrieval_scheme 10 3000 \
    --evaluation_metric_ks 1 10 50

elif [ ${1} = mitstates ]; then
  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/test.json \
    --evaluation_retrieval_scheme 10 0 \
    --evaluation_metric_ks 1 5 10 \
    --optimization_split_paths data/train.json \
    --backbone_name ${2} \
    --code_size 512 \
    --overlap_loss_weight 0.3 \
    --initial_temperature 1.0 \
    --use_gated_fusion True \
    --use_heuristic_negative True \
    --backbone_activity 1e-3 \
    --learning_rate 1e-4 \
    --warmup_epoch_count 6 \
    --total_epoch_count 64 \
    --batch_size 64 \
    --tcm_loss_weight 0.2 \
    --itm_loss_weight 0.1

  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/test.json \
    --evaluation_retrieval_scheme 10 0 \
    --evaluation_metric_ks 1 5 10

elif [ ${1} = css ]; then
  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/test.json \
    --evaluation_retrieval_scheme 1 0 \
    --evaluation_metric_ks 1 \
    --optimization_split_paths data/train.json \
    --backbone_name ${2} \
    --code_size 512 \
    --overlap_loss_weight 0.3 \
    --initial_temperature 1.0 \
    --use_gated_fusion True \
    --use_heuristic_negative True \
    --backbone_activity 1e-3 \
    --learning_rate 1e-4 \
    --warmup_epoch_count 6 \
    --total_epoch_count 64 \
    --batch_size 64 \
    --tcm_loss_weight 0.2 \
    --itm_loss_weight 0.1

  python ${ROOT}/main.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/test.json \
    --evaluation_retrieval_scheme 1 0 \
    --evaluation_metric_ks 1

else
  exit 1
fi
