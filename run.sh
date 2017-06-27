wget https://www.dropbox.com/s/zp91rycajh0lp4z/model-2380.data-00000-of-00001?dl=0
mv model-2380.data-00000-of-00001?dl=0 models/model-2380.data-00000-of-00001
python2.7 python/eval.py $1 $2