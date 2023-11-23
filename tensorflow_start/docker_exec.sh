echo $PWD

docker run -it --rm -v C:/Users/kirin/Desktop/AI/tesor-flow/tensorflow_start/:/tmp   tensorflow/tensorflow:latest-gpu \
    python ./tmp/$1
#bash
#   python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([10, 10])))"


#docker run -it --rm -v C:/Users/kirin/Desktop/AI/tesor-flow/tensorflow_start/:/tmp   tensorflow/tensorflow \
#    python ./tmp/defining_constant.py