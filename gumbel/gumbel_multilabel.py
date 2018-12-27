import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def sample_gumbel(shape, eps=1e-20, cuda = False):
    U = torch.rand(shape)
    temp = -torch.log(-torch.log(U+eps) + eps)
    if cuda:
        temp = temp.cuda()
    return temp

def gumbel_softmax_sample(logits, temperature, cuda=False):
    # logits : [bs, 2, n_labels]
    temp = Variable(sample_gumbel(logits.size(), cuda=cuda))
    if cuda:
        temp = temp.cuda()
    y = logits + temp
    return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, temperature, hard=False, cuda=False):
    y = gumbel_softmax_sample(logits, temperature, cuda=cuda)
    # y: [bs, 2, n_labels]
    if hard:
        k = y.size(-1)
        #    ported from Tensorflow
        #    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        #    y = tf.stop_gradient(y_hard - y) + y
        max_val, _ = torch.max(y, y.dim()-2,keepdim=True)
        #print(max_val)
        y_hard = (y==max_val).expand(y.size())
        y_hard = y_hard.type(torch.FloatTensor)
        if cuda:
            y_hard = y_hard.cuda()
        #y_hard = Variable(y_hard)
        y_new = (y_hard - y).detach() + y

        return y_new
    return y