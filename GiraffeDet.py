from json.tool import main
import tensorflow.keras.layers as k_l
import tensorflow.keras.models as k_model
from tensorflow.keras.activations import swish
from tensorflow.nn import space_to_depth
from tensorflow.image import resize

class upsample_like(k_l.Layer):
    def __init__(self,inter_type = 'bilinear' ,**kwargs):
        super(upsample_like,self).__init__(**kwargs)
        self.inter_type = inter_type
    
    def call(self, inputs, **kwargs):
        src,dst = inputs
        dst_b,dst_h,dst_w,dst_c = dst.shape.as_list()
        return resize(src,(dst_h,dst_w),self.inter_type)
    
    def compute_output_shape(self, input_shape):
        src_shape,dst_shape = input_shape
        return (src_shape[0],dst_shape[1],dst_shape[2],src_shape[3])
    



def _depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1):
  
    channel_axis = -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = k_l.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_{}'.format(block_id))(
            inputs)
        x = k_l.DepthwiseConv2D((3, 3),
                                    padding='same' if strides == (1, 1) else 'valid',
                                    depth_multiplier=depth_multiplier,
                                    strides=strides,
                                    use_bias=False,
                                    name='conv_dw_{}'.format(block_id))(
                                        x)
        x = k_l.BatchNormalization(
            axis=channel_axis, name='conv_dw_{}_bn'.format(block_id))(
                x)
        k_l.Activation(swish, name='conv_dw_{}_swish'.format(block_id))(x)

        x = k_l.Conv2D(
            pointwise_conv_filters, (1, 1),
            padding='same',
            use_bias=False,
            strides=(1, 1),
            name='conv_pw_{}'.format(block_id))(
                x)
        x = k_l.BatchNormalization(
            axis=channel_axis, name='conv_pw_{}_bn'.format(block_id))(
                x)
    return k_l.Activation(swish, name='conv_pw_{}_swish'.format(block_id))(x)

def conv2d_block(x,filters,k_size,strides=1,padding='same',bn=True,act=None,name=''):
    x = k_l.Conv2D(filters,k_size,strides=strides,padding=padding,
    name='{}_conv2d'.format(name))(x)
    if bn:
        x = k_l.BatchNormalization(axis=-1,epsilon=1e-5,
        name='{}_bn'.format(name))(x)
    if act:
        x = k_l.Activation(act,name='{}_act'.format(name))(x)

    return x


def _s2d_block(x):
    b,h,w,c = x.shape.as_list()
    padding_h = h%2
    padding_w = w%2

    x = k_l.ZeroPadding2D(((0,padding_h),(0,padding_w)))(x)
    x = k_l.Lambda(lambda x:space_to_depth(x,2))(x)
    return x



def s2d_chain(input_shape=(224,224,3),level=5):
    if isinstance(input_shape,list) or isinstance(input_shape,tuple):
        input_images = k_l.Input(input_shape)
    else:
        input_images = input_shape
    filters_init = 32
    x = conv2d_block(input_images,filters_init,3,2,bn=True,act=swish,name='b1')

    x = conv2d_block(x,filters_init*2,3,2,bn=True,act=swish,name='b2')

    x = _s2d_block(x)
    x = conv2d_block(x,filters_init*4,1,1,bn=True,act=swish,name='b3')

    x = _s2d_block(x)
    x = conv2d_block(x,filters_init*8,1,1,bn=True,act=swish,name='b4')

    x = _s2d_block(x)
    x = conv2d_block(x,filters_init*16,1,1,bn=True,act=swish,name='b5')

    if level>3:
        x = _s2d_block(x)
        x = conv2d_block(x,filters_init*32,1,1,bn=True,act=swish,name='b6')
    if level>4:
        x = _s2d_block(x)
        x = conv2d_block(x,filters_init*32,1,1,bn=True,act=swish,name='b7')

    model = k_model.Model(input_images,x)
    model.summary(line_length=150)

    return model

def queen_fusion_up(f_up,f_up_1,f_cur,f_down,filters,name = ''):
    if not f_up is None:
        f_up = conv2d_block(f_up,filters,1,1,bn=True,act=swish,name=name+'_u')
        # f_up = k_l.UpSampling2D(interpolation='bilinear',name=name+'_up_1')(f_up)
        f_up = upsample_like()((f_up,f_cur))
    if not f_up_1 is None:
        f_up_1 = conv2d_block(f_up_1,filters,1,1,bn=True,act=swish,name=name+'_u1')
        # f_up_1 = k_l.UpSampling2D(interpolation='bilinear',name=name+'_up_2')(f_up_1)
        f_up_1 = upsample_like()((f_up_1,f_cur))

    f_cur = conv2d_block(f_cur,filters,1,1,bn=True,act=swish,name=name+'_c')
    
    if not f_down is None:
        f_down = conv2d_block(f_down,filters,1,1,bn=True,act=swish,name=name+'_d')
        f_down = k_l.MaxPooling2D(pool_size=3,strides=2,padding='same',name=name+'_mp')(f_down)
    list_features = [f_down,f_cur,f_up,f_up_1]
    list_features = [f for f in list_features if not f is None]
    return k_l.Concatenate(axis=-1,name=name+'concate')(list_features)

def queen_fusion_down(f_up,f_cur,f_down,f_down_1,filters,name=''):
    if not f_up is None:
        f_up = conv2d_block(f_up,filters,1,1,bn=True,act=swish,name=name+'_u')
        # f_up = k_l.UpSampling2D(interpolation='bilinear',name=name+'_up')(f_up)
        f_up = upsample_like()((f_up,f_cur))

    f_cur = conv2d_block(f_cur,filters,1,1,bn=True,act=swish,name=name+'_c')

    if not f_down is None:
        f_down = conv2d_block(f_down,filters,1,1,bn=True,act=swish,name=name+'_d')
        f_down = k_l.MaxPooling2D(pool_size=3,strides=2,padding='same',name=name+'_mp_1')(f_down)

    if not f_down_1 is None:
        f_down_1 = conv2d_block(f_down_1,filters,1,1,bn=True,act=swish,name=name+'_d1')
        f_down_1 = k_l.MaxPooling2D(pool_size=3,strides=2,padding='same',name=name+'_mp_2')(f_down_1)
    list_features = [f_down,f_down_1,f_cur,f_up]
    list_features = [f for f in list_features if not f is None]
    return k_l.Concatenate(name=name+'concate')(list_features)

def skip_connect(skip_features,layer_index,depth,filters,name=''):
    log2features = []
    i = 0
    while layer_index-2**i>=0:
        log2features.append(skip_features[layer_index-2**i])
        i+=1
    log2feature = k_l.Concatenate(axis=-1,name=name)(log2features)
    x = k_l.SeparableConv2D(filters,3,padding='same')(log2feature)
    x = k_l.BatchNormalization(axis=-1,momentum=0.9,epsilon=1e-4)(x)
    x = k_l.Activation(swish)(x)

    return x

def _gfpn(features,depth=3,num_levels = 3,filters=96):
    """
    args:
    features  :backbone各分辨率特征图的list,分辨率由低到高
    depth     :GFPN的深度
    num_levels:FPN级数
    filters   :kernel size数量
    """
    log2skip_connect = [[feature] for feature in features]
    for i in range(1,depth):
        f_up_1 = None
        f_down_1 = None
        #P7->P3上采样阶段
        if i%2!=0:           
            for j in range(num_levels):  
                #print(i,j)     
                #最低分辨率特殊处理
                if j==0:
                    f_up = None
                else: 
                    if i==1:
                        f_up = features[j-1]
                    else:
                        f_up = log2skip_connect[j-1][-2]
                #最高分辨率特殊处理
                if j==(num_levels-1):
                    f_down = None
                else:
                    if i==1:
                        f_down =  features[j+1]
                    else:
                        f_down =  log2skip_connect[j+1][-1]
                #log2 skip connection
                if i==1:
                    f_cur = features[j]
                    # log2skip_connect[j].append(f_cur)
                else:
                    f_cur = skip_connect(
                        log2skip_connect[j],i,depth,filters,name='sc_{}_{}'.format(i,j))

                f_cur = queen_fusion_up(f_up,f_up_1,f_cur,f_down,filters,
                name = 'qf_{}_{}'.format(i,j))
                f_up_1 = f_cur
                log2skip_connect[j].append(f_cur)
                #print(log2skip_connect[j][-1].shape.as_list())
        #P3->P7下采样阶段
        else:
            for j in range(num_levels-1,-1,-1): 
                #print(i,j)
                #最低分辨率特殊处理
                f_up = None if j==0 else log2skip_connect[j-1][-1]
                #最高分辨率特殊处理
                f_down = None if j==(num_levels-1) else log2skip_connect[j+1][-2]

                f_cur = skip_connect(log2skip_connect[j],i,depth,filters,name='sc_{}_{}'.format(i,j))
                f_cur = queen_fusion_down(f_up,f_cur,f_down,f_down_1,filters,
                name = 'qf_{}_{}'.format(i,j))
                f_down_1 = f_cur
                log2skip_connect[j].append(f_cur)
                #print(log2skip_connect[j][-1].shape.as_list())
    outputs = []
    for i in range(num_levels):
        feature_project = conv2d_block(
            log2skip_connect[i][-1],96,1,act=swish,name='out_{}'.format(i)) 
        outputs.append(feature_project)
    return outputs



def GirafeeDet(input_shape=(224,224,3)):
    input_images = k_l.Input(input_shape)
    backbone = s2d_chain(input_images)
    feature_names = ['b7_act','b6_act','b5_act','b4_act','b3_act']
    layer_outputs = [backbone.get_layer(name).output for name in feature_names]

    backbone = k_model.Model(input_images,layer_outputs,name = backbone.name)

    features = _gfpn(backbone.outputs,7,5)

    g_det = k_model.Model(input_images,features)
    g_det.summary(line_length=150)
    print(g_det.outputs)

if __name__=='__main__':
    GirafeeDet((215,256,3))



        

