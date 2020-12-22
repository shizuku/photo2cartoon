import tensorflow as tf
import random
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

A_FILENAME = "raw_photo/{}.jpg"
B_FILENAME = "raw_cartoon/{}.jpg"

TRAIN_A_FILENAME = os.path.join(os.path.dirname(__file__), "train_a.tfrecord")
TRAIN_B_FILENAME = os.path.join(os.path.dirname(__file__), "train_b.tfrecord")
TEST_A_FILENAME = os.path.join(os.path.dirname(__file__), "test_a.tfrecord")
TEST_B_FILENAME = os.path.join(os.path.dirname(__file__), "test_b.tfrecord")

IMG_W = 256
IMG_H = 256
CHANNELS = 3


def serialize(tensor: tf.Tensor) -> str:
    return tf.io.serialize_tensor(tensor)


def parse(s: str) -> tf.Tensor:
    return tf.reshape(tf.io.parse_tensor(s, out_type=tf.float32), (1, IMG_W, IMG_H, CHANNELS))


def process(s: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tf.divide(tf.add(tf.cast(s, dtype=tf.float32), -127.5), 127.5), (1, IMG_W, IMG_H, CHANNELS))


AMOUNT = 2
DIV = 1


def cut(a: tf.Tensor) -> list:
    ie = a.shape[0] - IMG_W
    je = a.shape[1] - IMG_H
    r = []
    for _ in range(AMOUNT):
        i = random.randint(0, ie)
        j = random.randint(0, je)
        r.append(a[i:i + IMG_W, j:j + IMG_H])
    return r


def cut2(a: tf.Tensor, b: tf.Tensor) -> (list, list):
    ie = a.shape[0] - IMG_W
    je = a.shape[1] - IMG_H
    ra = []
    rb = []
    for _ in range(AMOUNT):
        i = random.randint(0, ie)
        j = random.randint(0, je)
        xa = a[i:i + IMG_W, j:j + IMG_H]
        xb = b[i:i + IMG_W, j:j + IMG_H]
        print(xa.shape, xb.shape)
        ra.append(xa)
        rb.append(xb)
    return ra, rb


def load_images(path_a: list, path_b: list) -> (list, list, list, list):
    train_a = []
    train_b = []
    test_a = []
    test_b = []
    for a, b in zip(path_a, path_b):
        img_a = tf.image.decode_image(tf.io.read_file(a))
        img_b = tf.image.decode_image(tf.io.read_file(b))
        ra = cut(img_a)
        rb = cut(img_b)
        # ra, rb = cut2(img_a, img_b)
        train_a.extend(ra[DIV:])
        train_b.extend(rb[DIV:])
        test_a.extend(ra[:DIV])
        test_b.extend(rb[:DIV])
    return train_a, train_b, test_a, test_b


def generate_data():
    train_a, train_b, test_a, test_b = load_images(
        [A_FILENAME.format(x) for x in range(20)],
        [B_FILENAME.format(x) for x in range(20)])

    train_a_ds = tf.data.Dataset.from_tensor_slices(train_a)
    train_b_ds = tf.data.Dataset.from_tensor_slices(train_b)
    train_a_ds = train_a_ds.map(process, num_parallel_calls=AUTOTUNE)
    train_b_ds = train_b_ds.map(process, num_parallel_calls=AUTOTUNE)
    train_a_ds = train_a_ds.map(serialize, num_parallel_calls=AUTOTUNE)
    train_b_ds = train_b_ds.map(serialize, num_parallel_calls=AUTOTUNE)
    train_a_ds_writer = tf.data.experimental.TFRecordWriter(TRAIN_A_FILENAME)
    train_a_ds_writer.write(train_a_ds)
    train_b_ds_writer = tf.data.experimental.TFRecordWriter(TRAIN_B_FILENAME)
    train_b_ds_writer.write(train_b_ds)

    test_a_ds = tf.data.Dataset.from_tensor_slices(test_a)
    test_b_ds = tf.data.Dataset.from_tensor_slices(test_b)
    test_a_ds = test_a_ds.map(process, num_parallel_calls=AUTOTUNE)
    test_b_ds = test_b_ds.map(process, num_parallel_calls=AUTOTUNE)
    test_a_ds = test_a_ds.map(serialize, num_parallel_calls=AUTOTUNE)
    test_b_ds = test_b_ds.map(serialize, num_parallel_calls=AUTOTUNE)
    test_a_ds_writer = tf.data.experimental.TFRecordWriter(TEST_A_FILENAME)
    test_a_ds_writer.write(test_a_ds)
    test_b_ds_writer = tf.data.experimental.TFRecordWriter(TEST_B_FILENAME)
    test_b_ds_writer.write(test_b_ds)
    print("writen", "train:", len(train_a), ", test:", len(test_a))


def load_train_data() -> (tf.data.Dataset, tf.data.Dataset):
    train_a_ds = tf.data.TFRecordDataset(TRAIN_A_FILENAME)
    train_b_ds = tf.data.TFRecordDataset(TRAIN_B_FILENAME)
    train_a_ds = train_a_ds.map(parse, num_parallel_calls=AUTOTUNE)
    train_b_ds = train_b_ds.map(parse, num_parallel_calls=AUTOTUNE)
    return train_a_ds, train_b_ds


def load_test_data() -> (tf.data.Dataset, tf.data.Dataset):
    test_a_ds = tf.data.TFRecordDataset(TEST_A_FILENAME)
    test_b_ds = tf.data.TFRecordDataset(TEST_B_FILENAME)
    test_a_ds = test_a_ds.map(parse, num_parallel_calls=AUTOTUNE)
    test_b_ds = test_b_ds.map(parse, num_parallel_calls=AUTOTUNE)
    return test_a_ds, test_b_ds


if __name__ == '__main__':
    generate_data()
