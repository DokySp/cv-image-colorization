
rm -fr datasets/

mkdir datasets
mkdir datasets/train
mkdir datasets/val

wget -P datasets https://brickcraft.doky.space/CV/train.tar.gz
wget -P datasets https://brickcraft.doky.space/CV/val.tar.gz
wget -P datasets https://brickcraft.doky.space/CV/test.tar.gz

tar -zxvf ./datasets/train.tar.gz -C ./datasets/train
tar -zxvf ./datasets/val.tar.gz -C ./datasets/val
tar -zxvf ./datasets/test.tar.gz -C ./datasets

rm datasets/train.tar.gz
rm datasets/val.tar.gz
rm datasets/test.tar.gz

rm datasets/.*
rm datasets/hint/.*
rm datasets/mask/.*
rm datasets/train/.*
rm datasets/val/.*

echo "Done!"