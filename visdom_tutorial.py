import visdom
import numpy as np
import random
viz = visdom.Visdom()
viz.close()
viz.image(
         np.random.rand(3, 512, 256), win='image',
         opts=dict(title='Random!', caption='How random.'),)
viz.line(X=np.array([0]),Y=np.array([0]),win="test",name='Line1',)
viz.line(X=np.array([0]),Y=np.array([0]),win="test2",name='Line1',)
i = 0
while(1):
    i += 1
    viz.image(np.random.rand(3, 512, 256), win='image', opts=dict(title='Random!!!!!'),)
    viz.line(X=np.array([i + 1]), Y=np.array([random.random()*10]), win="test", name='Line1', update='append',)
    viz.line(X=np.array([i + 1]), Y=np.array([random.random()*10]), win="test2", name='Line1', update='append',)