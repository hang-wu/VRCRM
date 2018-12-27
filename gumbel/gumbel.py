import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U+eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + Variable(sample_gumbel(logits.size()))
    return F.softmax(y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits, temperature)

    if hard:
        k = y.size(-1)

        #    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        #    y = tf.stop_gradient(y_hard - y) + y
        max_val, _ = torch.max(y, y.dim()-1,keepdim=True)
        #print(max_val)
        y_hard = (y==max_val).expand(y.size())
        y_hard = y_hard.type(torch.FloatTensor)
        #y_hard = Variable(y_hard)
        y_new = (y_hard - y).detach() + y

        return y_new
    return y