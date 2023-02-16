import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show

iris = load_iris()
X = iris.data
y = iris.target

clf = LogisticRegression(random_state=0, multi_class='multinomial')
clf.fit(X, y)

plots = []
for i in range(4):
    for j in range(i+1, 4):
        p = figure(title=f"Class distribution by {iris.feature_names[i]} and {iris.feature_names[j]}",
                   x_axis_label=iris.feature_names[i],
                   y_axis_label=iris.feature_names[j])
        classes = [0, 1, 2]
        colors = ['red', 'green', 'blue']
        for c, color in zip(classes, colors):
            idx = np.where(y == c)[0]
            p.circle(X[idx, i], X[idx, j], color=color, alpha=0.5, legend_label=f"Class {c}")

        xx, yy = np.meshgrid(np.linspace(X[:, i].min(), X[:, i].max(), 100),
                             np.linspace(X[:, j].min(), X[:, j].max(), 100))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        p.multi_line(xs=[xx[Z == c] for c in classes],
                     ys=[yy[Z == c] for c in classes],
                     color=colors,
                     line_width=2)
        plots.append(p)

show(gridplot(plots, ncols=2))
