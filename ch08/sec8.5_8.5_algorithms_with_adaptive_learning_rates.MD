## 8.5 Algorithms with Adaptive Learning Rates

### 8.5.1 AdaGrad

This is class Adagrad in [keras/optimizers.py](https://github.com/keras-team/keras/blob/master/keras/optimizers.py). 
The main different with previous section is LR is adapted by the whole gradient history:

```python
        for p, g, a in zip(params, grads, accumulators):
            new_a = a + K.square(g)  # update accumulator
            self.updates.append(K.update(a, new_a))
            new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)
```


***
### 8.5.2 RMSProp
class RMSprop in [keras/optimizers.py](https://github.com/keras-team/keras/blob/master/keras/optimizers.py). 
RMSprop modified the gradient accumulation into an exponentially weighted moving average.

```python
        for p, g, a in zip(params, grads, accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)
```


***
### 8.5.3 Adam

Class Adam in [keras/optimizers.py](https://github.com/keras-team/keras/blob/master/keras/optimizers.py). 
The book mentioned a few important distinctions with RMSProp. *m_t* refers to *s* and *v_t* refers to *r* in Algorithm 8.7 in the book.

```python
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t
```

***
Other optimizers including Adadelta, Adamax and Nadam exist in the file. Also refer [Keras doc](https://keras.io/optimizers/).
