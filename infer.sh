ROOT=$(dirname $(realpath ${0}))

source ${ROOT}/anaconda/bin/activate ${ROOT}/anaconda

if [ ${1} = fashioniq ]; then
  python ${ROOT}/infer.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/dress_val.json data/shirt_val.json data/toptee_val.json

elif [ ${1} = fashion200k ]; then
  python ${ROOT}/infer.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/all_test.json

elif [ ${1} = mitstates ]; then
  python ${ROOT}/infer.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/test.json

elif [ ${1} = css ]; then
  python ${ROOT}/infer.py \
    --image_base_path data/image_base \
    --evaluation_split_paths data/test.json

else
  exit 1
fi
