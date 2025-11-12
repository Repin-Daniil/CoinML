 for folder in 1 2 3 4 5; do
  aws s3 sync s3://perception-coins/coins/dataset/$folder ./dataset/$folder \
    --endpoint-url=https://storage.yandexcloud.net --profile yandex
  zip -r ${folder}.zip ./dataset/$folder
done