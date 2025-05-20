import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lr1', default=1e-4, type=float, help='learning rate for denoise model')
    parser.add_argument('--batch', default=2048, type=int, help='batch size')
    parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
    parser.add_argument('--reg', default=0.001, type=float, help='weight decay regularizer') 
    parser.add_argument('--ssl_reg_uu_ii', default=0.2, type=float, help='uni_uu_ii regularizer')
    parser.add_argument('--ssl_reg_ui', default=0.3, type=float, help='uni_ui regularizer')
    parser.add_argument('--temperature', default=5, type=float, help='uni_temperature')
    parser.add_argument('--temperature1', default=0.5, type=float, help='u-i_uni_temperature')
    parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
    parser.add_argument('--decay', default=0, type=float, help='weight decay rate')
    parser.add_argument('--latdim', default=300, type=int, help='embedding size')
    parser.add_argument('--mask_r', default=1, type=float, help='mask ratio')
    parser.add_argument('--lp', default=0, type=float, help='mask ratio low bound')
    parser.add_argument('--gcn_layer0', default=2, type=int, help='number of gcn layers')



    # diffusion model
    parser.add_argument('--noise_scale', default=0.01, type=float, help='the scale of noise')
    parser.add_argument('--noise_min', default=0.005, type=float, help='lower bounds of the added noises')
    parser.add_argument('--noise_max', default=0.01, type=float, help='upper bounds of the added noises')
    parser.add_argument('--time_step', default=80, type=int, help='diffusion step')
    parser.add_argument('--sample_step', default=0, type=int, help='p_sample: denoise step')
    parser.add_argument('--noiseDirection', default=True, type=bool, help='noiseDirection')
    parser.add_argument('--sampleNoise', default=False, type=bool, help='sample noise')
    parser.add_argument('--elbo', default=0.01, type=float, help='diffusion regularizer')



    # denoise model
    parser.add_argument('--mlp_dims', default='[600]', type=str, help='denoise model hidden dim')
    parser.add_argument('--emb_size', default=10, type=float, help='time emb size')
    parser.add_argument('--actFunc', default='tanh', type=str, help='activation function')
    parser.add_argument('--norm', default=False, type=bool, help='denoise normalize')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout')
    parser.add_argument('--dropout1', default=0, type=float, help='dropout')
    parser.add_argument('--residual', default=False, type=bool, help='residual net in denoise model')
    parser.add_argument('--sample_methon', default="importance", type=str, help='assign different weight to different timestep or not')
    parser.add_argument('--scale', default=0.2, type=float, help='weight of uncondition to condition')



    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--data', default='ml-1m', type=str, help='name of dataset') # ml-1m yelp douban
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
    parser.add_argument('--seed', default='3407', type=int, help='model seed')

    return parser.parse_args()


args = ParseArgs()