wget 'https://s3.amazonaws.com/fast-ai-sample/mnist_sample.tgz'
tar zxf mnist_sample.tgz
rm mnist_sample.tgz

mkdir mnist_sample/3
mv mnist_sample/train/3/* mnist_sample/3

mkdir mnist_sample/7
mv mnist_sample/train/7/* mnist_sample/7

rm -rf mnist_sample/train mnist_sample/valid mnist_sample/labels.csv