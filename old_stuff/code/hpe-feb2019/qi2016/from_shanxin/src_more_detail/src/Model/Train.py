__author__ = 'QiYE'

import theano.tensor as T
import theano
import numpy
def update_params(params,grads,gamma = 0.0,
    lamda = 0.01,
    yita = 0.000):

    delta = []
    for param_i in params:
        delta.append(theano.shared(param_i.get_value(), borrow=False))
    updates = []
    for param_i, delta_i, grad_i in zip(params, delta, grads):
        updates.append((delta_i, gamma*delta_i - lamda*(yita*param_i + grad_i)))
        updates.append((param_i, param_i + gamma*delta_i - lamda*(yita*param_i+grad_i)))
    return updates

def update_params2(model,cost,learning_rate,momentum,):

    updates = []

    for param in  model.params:
      param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))
      updates.append((param, param - learning_rate*param_update))
      updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param)))

    return updates
def adam_update(model,cost,m,v, learning_rate,beta1,beta2,beta1_t,beta2_t):

    updates = []

    for param_i,m_i,v_i in  zip(model.params,m,v):
        g = T.grad(cost,param_i)
        tmp_m_i = (beta1*m_i+(1-beta1)*g)
        tmp_v_i = (beta2*v_i+(1-beta2)*T.square(g))
        updates.append((m_i, tmp_m_i))
        updates.append((v_i, tmp_v_i))
        updates.append((param_i, param_i - learning_rate*tmp_m_i/(1-beta1_t)/(T.sqrt(tmp_v_i/ (1-beta2_t))+0.00000001)))
    return updates



def set_params_initial(params):
    for param_i in (params):
        param_i.set_value(0)
    return

def set_params(path,params):
    model_info = numpy.load(path)
    params_val = model_info[0]
    # print 'model trained', model_info[1]
    # cost_v = model_info[-1][-1]
    # print 'cost_v',cost_v
    for param_i, params_v in zip(params, params_val):
        param_i.set_value(params_v)
    return

def set_adam_params(path,params):
	model_info = numpy.load(path)
	params_val = model_info[0]
	m_val = model_info[1]
	v_val = model_info[2]
	beta1 = model_info[3]
	beta2 = model_info[4]
	train_cost=model_info[-2]
	test_cost=model_info[-1]

	for param_i, params_v in zip(params, params_val):
		param_i.set_value(params_v)

	m=[]
	v=[]
	for m_i,v_i in  zip(m_val,v_val):
		m.append(theano.shared(numpy.cast[theano.config.floatX](m_i)))
		v.append(theano.shared(numpy.cast[theano.config.floatX](v_i)))
		beta1_t=theano.shared(numpy.cast[theano.config.floatX](beta1), name='momentum')
		beta2_t=theano.shared(numpy.cast[theano.config.floatX](beta2), name='momentum')
	return m,v,beta1_t,beta2_t,train_cost,test_cost

def get_gradients(model,cost):

    return T.grad(cost, model.params)
