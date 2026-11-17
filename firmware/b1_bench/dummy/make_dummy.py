import numpy as np, tensorflow as tf
tf.keras.utils.set_random_seed(0)
T, C = 200, 8   # 4 s @ 50 Hz; acc xyz, gyro xyz, |acc|, |gyro|
def ds_block(x, f, d):
    y = tf.keras.layers.DepthwiseConv1D(5, padding="same", dilation_rate=d)(x)
    y = tf.keras.layers.Conv1D(f, 1)(y)
    y = tf.keras.layers.BatchNormalization()(y); y = tf.keras.layers.ReLU()(y)
    if x.shape[-1] == f: y = tf.keras.layers.Add()([x, y])
    return y
i = tf.keras.Input((T, C))
x = tf.keras.layers.Conv1D(32, 5, strides=2, padding="same", activation="relu")(i)
for d in (1, 2, 4, 8): x = ds_block(x, 32, d)
x = tf.keras.layers.Conv1D(48, 1, activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
o = tf.keras.layers.Dense(2)(x)
m = tf.keras.Model(i, o); print("params", m.count_params())
X = np.random.default_rng(0).normal(size=(256, T, C)).astype("float32")
m.compile("adam", tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
m.fit(X, (X[:, :, 0].std(1) > 1).astype(int), epochs=1, verbose=0)
cv = tf.lite.TFLiteConverter.from_keras_model(m)
cv.optimizations = [tf.lite.Optimize.DEFAULT]
cv.representative_dataset = lambda: ([X[k:k+1]] for k in range(100))
cv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
cv.inference_input_type = tf.int8; cv.inference_output_type = tf.int8
open("dummy_dstcn_int8.tflite", "wb").write(cv.convert())
np.save("dummy_inputs_f32.npy", X[:50])
