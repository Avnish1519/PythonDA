from sklearn.linear_model import LinearRegression
import numpy as np

x=np.array([[1],[2],[3],[4],[5]])
y=np.array([50,55,65,70,75])

model = LinearRegression()
model.fit(x,y)

prediction=model.predict([[6]])
print(f"Predicted score for 6 hours:{prediction[0]:2f}")