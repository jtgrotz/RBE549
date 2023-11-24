import keras
import numpy as np

#crate layers with 1 input that outputs one output
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
#choose loss funciton
model.compile(optimizer='sgd', loss='mean_squared_error')

#initialize data
xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype=float)
ys = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0], dtype=float)

#create model
model.fit(xs,ys,epochs=500)

#run model
print(model.predict([10.0]))

