 for folder in 1 2 3 4 5; do
  wget https://storage.yandexcloud.net/perception-coins/coins/dump/${folder}.zip
  unzip ${folder}.zip
  rm ${folder}.zip
done