from src.utils import constants

__author__ = 'QiYE'

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
import numpy

def apply_dropout(input, rng, p=0.5):
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout resp. dropconnect is applied

    :type p: float or double between 0. and 1.
    :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.

    """
    mask = T.shared_randomstreams.RandomStreams(rng.randint(999999)).binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask


class ConvPoolLayer(object):
    """Pool Layer of a convolutional network """
    def __init__(self, layer_name, rng,is_train, input, filter_shape, image_shape, poolsize=(2, 2), sub_x= (1,1),activation="None",p=0.5):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            subsample=sub_x
        )
        lin_output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        """ Applies nonlinearity over the convolutional layer."""
        if activation == 'relu':
            lin_output = T.switch(lin_output > 0., lin_output, 0)
        if activation == 'tanh':
            lin_output = T.tanh(lin_output)

        # downsample each feature map individually, using maxpooling
        if poolsize[0]>1:
            pool_output = pool.pool_2d(
                input=lin_output,
                ds=poolsize,
                ignore_border=True
            )
        else:
            pool_output = lin_output


        # multiply output and drop -> in an approximation the scaling effects cancel out
        # train_output = apply_dropout(numpy.cast[theano.config.floatX](1./p) * pool_output,p=p,rng=rng)
        # #is_train is a pseudo boolean theano variable for switching between training and prediction
        # self.output = T.switch(T.neq(is_train, 0), train_output, pool_output)
        self.output=pool_output
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        # store parameters of this layer
        self.W.name = layer_name+'_W'
        self.b.name = layer_name+'_b'
        self.params = [self.W, self.b]


class FullConLayer(object):
    def __init__(self, layer_name, rng, is_train,input, n_in, n_out, W=None, b=None,
                 activation=None,p=0.5):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
                           could be 'relu','tanh', 'None'
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        if activation is None:
            act_output = lin_output
        else:
            if  activation == 'relu':
               act_output =  T.switch(lin_output > 0., lin_output, 0)
            else:
               act_output = T.tanh(lin_output)

        # train_output = apply_dropout(numpy.cast[theano.config.floatX](1./p) * act_output,p=p,rng=rng)
        # #is_train is a pseudo boolean theano variable for switching between training and prediction
        # self.output = T.switch(T.neq(is_train, 0), train_output, act_output)
        self.output =act_output

        self.W.name = layer_name+'_W'
        self.b.name = layer_name+'_b'
        self.params = [self.W, self.b]

    def cost(self, Y):
        diff = T.sqr(Y - self.output)
        cost_matrix = T.sum(T.reshape(diff,(diff.shape[0], constants.NUM_JNTS, constants.OUT_DIM )), axis=-1)
        return cost_matrix.mean(axis=1).mean()
        # return cost_matrix.sum(axis=1).mean()


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, layer_name, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class LogisticRegression_ClassGroup(object):

    def __init__(self, input, n_in, n_outs):
        """ Initialize the parameters of the logistic regression

        :type n_outs: list of int
        :param n_outs: number of output units in each group

        """
        n_out = numpy.sum(n_outs)
        n_groups = len(n_outs)
        self.n_groups = n_groups
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in,n_out), dtype = theano.config.floatX),
                                name='W')
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype = theano.config.floatX),
                               name='b')

        self.h = T.dot(input, self.W) + self.b
        self.p_y_given_x = []
        self.y_pred = []
        t = 0
        for idx in xrange(n_groups):
            p_y_given_x = T.nnet.softmax( self.h[t:t+n_outs[idx]])
            y_pred = T.argmax( p_y_given_x, axis = 1)
            t+= n_outs[idx]
            self.p_y_given_x.append( p_y_given_x)
            self.y_pred.append( y_pred )

        # parameters of the model
        self.params = [self.W, self.b]


    def negative_log_likelihood(self, ys):
        cost = -T.mean(T.log( self.p_y_given_x[0])[T.arange(ys[0].shape[0]),ys[0]])
        for idx in xrange(1, self.n_groups):
            cost += -T.mean(T.log( self.p_y_given_x[idx])[T.arange(ys[idx].shape[0]),ys[idx]])

        return cost


    def errors(self, ys):
        errs = []
        for idx in xrange(self.n_groups):
            if ys[idx].ndim != self.y_pred[idx].ndim:
                raise TypeError('y should have the same shape as self.y_pred',
                    ('y', ys[idx].type, 'y_pred', self.y_pred[idx].type))
            # check if y is of the correct datatype
            if ys[idx].dtype.startswith('int'):
                # the T.neq operator returns a vector of 0s and 1s, where 1
                # represents a mistake in prediction
                errs.append( T.mean(T.neq(self.y_pred[idx], ys[idx])))
            else:
                raise NotImplementedError()
        return errs
class Comp_Class_Regress(object):

    def __init__(self, input, n_in, n_outs):
        """ Initialize the parameters of the logistic regression

        :type n_outs: list of int,
        for example:
        when the model regresses the center of the palm and classifies the rotation whose label ranges from 0 to 180,
        the n_outs should be [3,180]
        :param n_outs: number of output units in each group

        """
        n_out = numpy.sum(n_outs)
        n_groups = len(n_outs)
        self.n_groups = n_groups
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in,n_out), dtype = theano.config.floatX),
                                name='W')
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype = theano.config.floatX),
                               name='b')

        self.h = T.dot(input, self.W) + self.b
        self.p_y_given_x = []
        self.y_pred = []
        t = 0
        self.line_out = self.h[:,t:t+n_outs[0]]
        t += n_outs[0]
        self.p_y_given_x = T.nnet.softmax( self.h[:,t:t+n_outs[1]])
        # parameters of the model
        self.params = [self.W, self.b]


    def cost(self, ys):
        diff = T.sqr(ys[0] - self.line_out)
        cost1 = T.sum(T.reshape(diff,(diff.shape[0], constants.NUM_JNTS, constants.OUT_DIM )), axis=-1).mean(axis=1).mean()

        cost2 = -T.mean(T.log( self.p_y_given_x)[T.arange(ys[1].shape[0]),ys[1]])

        return cost1+cost2, cost1, cost2



    def errors(self, ys):
        errs = []
        for idx in xrange(self.n_groups):
            if ys[idx].ndim != self.y_pred[idx].ndim:
                raise TypeError('y should have the same shape as self.y_pred',
                    ('y', ys[idx].type, 'y_pred', self.y_pred[idx].type))
            # check if y is of the correct datatype
            if ys[idx].dtype.startswith('int'):
                # the T.neq operator returns a vector of 0s and 1s, where 1
                # represents a mistake in prediction
                errs.append( T.mean(T.neq(self.y_pred[idx], ys[idx])))
            else:
                raise NotImplementedError()
        return errs

class CompLayer(object):
    def __init__(self,layer_name, num_layers, rng, input, is_train,n_in,p=0.5):
        self.num_layers = num_layers
        self.layer_name = layer_name
        self.layers = []
        self.output =[]
        self.params = []
        # for i in xrange(num_layers):
        layer = FullConLayer(
            layer_name='comp_lin_out',
            rng=rng,
            is_train=is_train,
            input=input,
            n_in= n_in,
            n_out=constants.NUM_JNTS* constants.OUT_DIM,
            activation=None,
            p=p)
        self.layers.append(layer)
        self.params += layer.params
        self.output.append(layer.output)

        layer = LogisticRegression(
            layer_name='comp_log_out',
            input=input,
            n_in= n_in,
            n_out=constants.Num_Class
        )

        self.layers.append(layer)
        self.output.append(layer.p_y_given_x)
        self.params += layer.params


    def cost(self, Y, Rot):
        cost=[]
        cost.append(self.layers[0].cost(Y))
        cost.append(self.layers[0].cost(Rot))
        return cost