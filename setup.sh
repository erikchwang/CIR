ROOT=$(dirname $(realpath ${0}))

wget -P ${ROOT} https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ${ROOT}/Miniconda3-latest-Linux-x86_64.sh -b -f -p ${ROOT}/anaconda
rm ${ROOT}/Miniconda3-latest-Linux-x86_64.sh

source ${ROOT}/anaconda/bin/activate ${ROOT}/anaconda
conda install -y python=3.9.13
conda install -y -c pytorch pytorch=1.11.0 cudatoolkit=10.2
pip install Pillow==9.1.1
pip install ray[tune]==1.13.0
pip install transformers==4.20.0

if [ ${1} = fashioniq ]; then
  wget -P ${ROOT} https://cmr-datasets.s3.us-east-2.amazonaws.com/fashioniq/fashioniq_data.zip
  unzip -d ${ROOT} ${ROOT}/fashioniq_data.zip
  rm ${ROOT}/fashioniq_data.zip

elif [ ${1} = fashion200k ]; then
  wget -P ${ROOT} https://cmr-datasets.s3.us-east-2.amazonaws.com/fashion200k/fashion200k_data.zip
  unzip -d ${ROOT} ${ROOT}/fashion200k_data.zip
  rm ${ROOT}/fashion200k_data.zip

elif [ ${1} = mitstates ]; then
  wget -P ${ROOT} https://cmr-datasets.s3.us-east-2.amazonaws.com/mitstates/mitstates_data.zip
  unzip -d ${ROOT} ${ROOT}/mitstates_data.zip
  rm ${ROOT}/mitstates_data.zip

elif [ ${1} = css ]; then
  wget -P ${ROOT} https://cmr-datasets.s3.us-east-2.amazonaws.com/css/css_data.zip
  unzip -d ${ROOT} ${ROOT}/css_data.zip
  rm ${ROOT}/css_data.zip

else
  exit 1
fi
