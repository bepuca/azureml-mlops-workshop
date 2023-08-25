wget 'https://s3.amazonaws.com/fast-ai-sample/mnist_sample.tgz'
tar zxf mnist_sample.tgz
rm mnist_sample.tgz

mkdir mnist_images
mkdir mnist_images/subset

mkdir mnist_images/subset/3
mv mnist_sample/train/3/* mnist_images/subset/3

mkdir mnist_images/subset/7
mv mnist_sample/train/7/* mnist_images/subset/7

rm -rf mnist_sample