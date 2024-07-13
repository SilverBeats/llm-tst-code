dataset=$1

if [ "$dataset" == "yelp" ]; then
  python baseline.py \
            --dataset "$dataset" \
            --acc_model_dir_or_path "output/classifier/pos-neg/yelp/classifier.pk" \
            --d_ppl_model_dir "output/fluency/pos-neg/yelp/4"
elif [ "$dataset" == "amazon" ]; then
  python baseline.py \
            --dataset "$dataset" \
            --acc_model_dir_or_path "output/classifier/pos-neg/amazon/classifier.pk" \
            --d_ppl_model_dir "output/fluency/pos-neg/amazon/4"
elif [ "$dataset" == "imagecaption" ]; then
  python baseline.py \
            --dataset "$dataset" \
            --acc_model_dir_or_path "output/classifier/romantic-humorous/imagecaption/classifier.pk" \
            --d_ppl_model_dir "output/fluency/romantic-humorous/imagecaption/4"
elif [ "$dataset" == "gender" ]; then
  python baseline.py \
            --dataset "$dataset" \
            --acc_model_dir_or_path "output/classifier/male-female/gender/classifier.pk" \
            --d_ppl_model_dir "output/fluency/male-female/gender/4"
elif [ "$dataset" == "political" ]; then
  python baseline.py \
            --dataset "$dataset" \
            --acc_model_dir_or_path "output/classifier/republican-democratic/political/classifier.pk" \
            --d_ppl_model_dir "output/fluency/republican-democratic/political/4"
else
  echo '未实现'
fi