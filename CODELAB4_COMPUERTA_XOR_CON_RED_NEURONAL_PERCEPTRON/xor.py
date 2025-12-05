import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input 

X = np.array([  
    [0, 0],  
    [0, 1],  
    [1, 0],  
    [1, 1]  
])  
y = np.array([  
    [0],  
    [1],  
    [1],  
    [0]  
])
model = Sequential([  
    Input(shape=(2,)),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid") 
])
model.compile(  
    optimizer="adam",  
    loss="binary_crossentropy",  
    metrics=["accuracy"]  
)
history = model.fit(  
    X, y,  
    epochs=5000,
    verbose=0
)
print("\nEvaluación:")  
loss, acc = model.evaluate(X, y, verbose=0)  
print(f"Loss: {loss:.3f}, Accuracy: {acc:.3f}")  
  
print("\nPredicciones XOR:")  
for a, b in X:  
    pred = model.predict(np.array([[a, b]]), verbose=0)  
    print(f"{a} XOR {b} = {round(pred.item(), 3)}")
