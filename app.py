from io import BytesIO
import io
from flask import Flask, render_template, request, send_file, g, jsonify, send_from_directory
import argparse
import os
import random
import cv2
import base64
from PIL import Image, ImageEnhance
import numpy as np
# engine
from sdr import StableDiffusionPipeline
from sdr import UpscaleModel


app = Flask(__name__, static_folder='dist/assets', static_url_path='/assets')
TEST=False
DEVICE_ID=os.environ.get('DEVICE_ID', 30)
BASENAME = os.environ.get('BASENAME', 'deliberate-lora_pixarStyleLora_lora128-unet-2')
CONTROLNET = os.environ.get('CONTROLNET', 'canny_multize')
RETURN_BASE64 = bool(int(os.environ.get('RETURN_BASE64', 1)))

SHAPES=[[640, 512], [192, 384], [448, 448], [512, 576], [384, 128], [384, 448], [448, 256], \
             [896, 512], [448, 128], [512, 384], [320, 512], [192, 512], [512, 640], [384, 320], \
                [384, 512], [704, 512], [512, 448], [128, 448], [320, 448], [256, 384], [384, 384], \
                    [128, 512], [512, 704], [320, 384], [832, 512], [512, 832], [448, 512], [512, 320], \
                        [256, 512], [448, 192], [512, 896], [256, 448], [128, 384], [384, 256], [576, 512], \
                            [768, 512], [512, 256], [512, 512], [384, 192], [512, 192], [512, 128], [768, 768], \
                                [512, 768], [192, 448], [448, 320], [448, 384]]



from flask_cors import CORS
CORS(app, supports_credentials=True)

def get_fixed_seed(seed):
    if seed is None or seed == '' or seed == -1:
        return int(random.randrange(4294967294))
    return seed

def fix_iter_seed(seed,iter):
    iter += 1
    # current iter standards for the current image in batch, which is not the same with webui
    assert iter >= 1
    iter = iter - 1
    delta_seed = 0 if iter == 0 else int(random.randrange(100000*iter))
    return seed + delta_seed

@app.before_first_request
def load_model():
    pipeline = StableDiffusionPipeline(
        basic_model=BASENAME,
        controlnet_name=CONTROLNET)
    app.config['pipeline'] = pipeline
    app.config['base_model'] = BASENAME
    app.config['available'] = True
    print("register pipeline to app object.")
    print('pipeline is in app.config:', 'pipeline' in app.config)


# 模型切换：如果 给生图请求加上basemodel参数，“允许用户的文生图、图生图”触发模型切换，那在run pipeline之前要先check模型对不对，不对就要重新load
# 但此时其他用户可能还在用当前模型，触发模型reload可能非常频繁
@app.route('/switch', methods=['POST'])
def switch_checkpoint():
    data = request.get_json()
    new_basemodel_name = data.get('basemodel_name')
    if new_basemodel_name != app.config['base_model']:
        app.config['available'] = False
        del app.config['pipeline']
        app.config['pipeline'] = StableDiffusionPipeline(basic_model=new_basemodel_name, controlnet_name=CONTROLNET)
        app.config['base_model'] = new_basemodel_name
        app.config['available'] = True
        res = {'results': "Model has been set to {}".format(app.config['base_model'])}
    else:
        res = {'results': "no changes"}
    # 设置响应头
    response = jsonify(res)
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route('/chlora', methods=['POST'])
def add_or_rm_lora():
    data = request.get_json()
    lora_state = data.get('withlora')
    changed = app.config['pipeline'].set_lora(lora_state)
    response = jsonify({'changed': changed, 'hasLora': lora_state})
    response.headers['Content-Type'] = 'application/json'
    return response


@app.before_request
def check_model():
    assert app.config['available'] == True


def handle_base64_image(controlnet_image):
    # 目前只支持一个controlnet_image, 不可以是list
    if isinstance(controlnet_image, list):
        controlnet_image = controlnet_image[0]
    if controlnet_image.startswith("data:image"):
        controlnet_image = controlnet_image.split(",")[1]
        
    return controlnet_image

def handle_output_base64_image(image_base64):
    if not RETURN_BASE64:
        return image_base64
    if not image_base64.startswith("data:image"):
        image_base64 = "data:image/jpeg;base64," + image_base64
    return image_base64

# TODO
def get_shape_by_ratio(width, height):
    ratio_shape = {
        1:[512,512],
        2/3:[640,960],
        3/2:[960,640],
        4/3:[704,896],
        3/4:[896,704],
        9/16:[576,1024],
        16/9:[1024,576],
    }
    ratio = width/height
    # 这个ratio找到最接近的ratio_shape
    ratio_shape_list = list(ratio_shape.keys())
    ratio_shape_list.sort(key=lambda x:abs(x-ratio))
    nshape = ratio_shape[ratio_shape_list[0]]
    print(nshape)
    return nshape


@app.route('/')
def home():
    return send_file('dist/index.html')

# 静态文件

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'dist'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route("/hello")
def hello():
    return "Hello World!"

@app.route('/txt2img', methods=['POST'])
def process_data():
    # 从请求中获取 JSON 数据
    data = request.get_json()
    # 从 JSON 数据中获取所需数据
    prompt = data.get('prompt')
    negative_prompt = data.get('negative_prompt')
    num_inference_steps = int(data.get('steps'))
    guidance_scale = int(data.get('cfg_scale', 7))
    strength = float(data.get('denoising_strength'))
    sampler_index = data.get('sampler_index', "Euler a")
    seed = get_fixed_seed(int(data.get('seed')))
    subseed = get_fixed_seed(int(data.get('subseed')))# 不可以为-1
    subseed_strength = float(data.get('subseed_strength'))
    seed_resize_from_h = data.get('seed_resize_from_h',1)
    seed_resize_from_w = data.get('seed_resize_from_w',1)
    n_iter = int(data.get('n_iter', 1))
    seed = fix_iter_seed(seed, n_iter)
    # data 是否包含 args的参数 
    controlnet_image = None
    flag = True
    controlnet_args = {}
    if 'alwayson_scripts' in data:
        if "controlnet" in data['alwayson_scripts']:
            if "args" in data['alwayson_scripts']['controlnet']:
                controlnet_args = data['alwayson_scripts']['controlnet']['args'][0]
                if "enabled" in data['alwayson_scripts']['controlnet']['args'][0]:
                    if data['alwayson_scripts']['controlnet']['args'][0]['enabled']==False:
                        controlnet_image= None
                        flag = False
                    else:
                        flag = True
                else:
                    flag = False
                    controlnet_image= None
                if len(data['alwayson_scripts']['controlnet']['args']) ==1  and flag:
                    args_info = data['alwayson_scripts']['controlnet']['args'][0]
                    if 'image' in args_info:
                        controlnet_image = data['alwayson_scripts']['controlnet']['args'][0]['image']
                        # base64 to image
                        controlnet_image = base64.b64decode(controlnet_image)
                        controlnet_image = Image.open(io.BytesIO(controlnet_image))
                else:
                    controlnet_image = None
    init_image = None
    mask = None

    with app.app_context():
        pipeline = app.config['pipeline']  # 获取 pipeline 变量
        width = int(data.get('width', 512))
        height = int(data.get('height', 512))
        assert [width, height] in SHAPES
        pipeline.set_height_width(height, width)
        try:
            pipeline.scheduler = sampler_index
            img_pil = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=init_image,
                mask=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_img = controlnet_image,
                seeds = [seed],
                subseeds = [subseed],
                subseed_strength=subseed_strength,
                seed_resize_from_h=seed_resize_from_h,
                seed_resize_from_w=seed_resize_from_w,
                controlnet_args = controlnet_args,
            )
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            print(e)
            print("error")

    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 构建JSON响应
    response = jsonify({'images': [ret_img_b64]})

    # 设置响应头
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route('/img2img', methods=['POST'])
def process_data_img():
    # 从请求中获取 JSON 数据
    data = request.get_json()
    # 从 JSON 数据中获取所需数据
    prompt = data.get('prompt')
    negative_prompt = data.get('negative_prompt')
    num_inference_steps = int(data.get('steps'))
    guidance_scale = int(data.get('cfg_scale'))
    strength = float(data.get('denoising_strength'))
    seed = get_fixed_seed(int(data.get('seed')))
    sampler_index = data.get('sampler_index', "Euler a")
    # data 是否包含 args的参数 
    controlnet_image = None
    init_image = None
    mask = None
    init_image_b64 = data['init_images'][0]
    mask_image_b64 = data.get('mask') or None
    subseed = get_fixed_seed(int(data.get('subseed')))# 不可以为-1
    subseed_strength = float(data.get('subseed_strength'))
    seed_resize_from_h = data.get('seed_resize_from_h',0)
    seed_resize_from_w = data.get('seed_resize_from_w',0)
    n_iter = int(data.get('n_iter', 1))
    seed = fix_iter_seed(seed, n_iter)
    width = int(data.get('width', 512))
    height = int(data.get('height', 512))


    if init_image_b64:
        init_image_bytes = BytesIO(base64.b64decode(init_image_b64))
        init_image = Image.open(init_image_bytes)
    if init_image_b64 and mask_image_b64:
        mask = BytesIO(base64.b64decode(mask_image_b64))
        mask[mask > 0] = 255
    else:
        mask = None

    controlnet_image = None
    flag = True
    controlnet_args  = {}
    if 'alwayson_scripts' in data:
        if "controlnet" in data['alwayson_scripts']:
            if "args" in data['alwayson_scripts']['controlnet']:
                controlnet_args = data['alwayson_scripts']['controlnet']['args'][0]
                if "enabled" in data['alwayson_scripts']['controlnet']['args'][0]:
                    if data['alwayson_scripts']['controlnet']['args'][0]['enabled']==False:
                        controlnet_image= None
                        flag = False
                    else:
                        flag = True
                else:
                    flag = False
                    controlnet_image= None
                if len(data['alwayson_scripts']['controlnet']['args']) ==1  and flag:
                    args_info = data['alwayson_scripts']['controlnet']['args'][0]
                    if 'image' in args_info:
                        controlnet_image = data['alwayson_scripts']['controlnet']['args'][0]['image']
                        # base64 to image
                        controlnet_image = base64.b64decode(controlnet_image)
                        controlnet_image = Image.open(io.BytesIO(controlnet_image))
                else:
                    controlnet_image = None

    with app.app_context():
        pipeline = app.config['pipeline']  # 获取 pipeline 变量
        assert [width, height] in SHAPES
        pipeline.set_height_width(height, width)
        try:
            pipeline.scheduler = sampler_index
            img_pil = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=init_image,
                mask=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_img = controlnet_image,
                seeds = [seed],
                subseeds = [subseed],
                subseed_strength=subseed_strength,
                seed_resize_from_h=seed_resize_from_h,
                seed_resize_from_w=seed_resize_from_w,
                controlnet_args = controlnet_args,
                use_controlnet = flag
            )
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            print(e)
            print("error")
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 构建JSON响应
    response = jsonify({'images': [ret_img_b64]})

    # 设置响应头
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/upscale", methods=['POST'])
def process_upscale():
    # =================================================#
    # 在upscale的时候 需要controlnetimg和initimg为同一张图
    # 但是为了传输方便 这里的controlnetimg可以为空 默认为原图
    # =================================================#
    # 从请求中获取 JSON 数据
    data = request.get_json()
    # 从 JSON 数据中获取所需数据
    prompt = data.get('prompt')
    negative_prompt = data.get('negative_prompt')
    num_inference_steps = int(data.get('steps'))
    guidance_scale = int(data.get('cfg_scale'))
    strength = float(data.get('denoising_strength'))
    seed = get_fixed_seed(int(data.get('seed')))
    controlnet_image = None
    init_image = None
    mask = None
    init_image_b64 = data['init_images'][0]
    mask_image_b64 = data.get('mask') or None
    subseed = get_fixed_seed(int(data.get('subseed')))# 不可以为-1
    subseed_strength = float(data.get('subseed_strength'))
    seed_resize_from_h = data.get('seed_resize_from_h',1)
    seed_resize_from_w = data.get('seed_resize_from_w',1)
    sampler_index = data.get('sampler_index', "Euler a")
    width =  512
    height = 512

    if init_image_b64:
        init_image_bytes = BytesIO(base64.b64decode(init_image_b64))
        init_image = Image.open(init_image_bytes)
    if init_image_b64 and mask_image_b64:
        mask = BytesIO(base64.b64decode(mask_image_b64))
        mask[mask > 0] = 255
    else:
        mask = None
    controlnet_image = None
    controlnet_args  = {}
    flag = True
    # upscale 参数处理 
    upscale_factor = int(data.get('upscale_factor', 2))# 必须大于0 且必须为整数
    target_width   = int(data.get('target_width', 1024))
    target_height  = int(data.get('target_height', 1024))
    # upscale和target必需传一个，两个都传的话以upscale_factor为准
    upscale_type   = data.get('upscale_type', 'LINEAR')# 必须大写 只有两种形式 LINEAR 和 CHESS
    # tile_width     = int(data.get('tile_width', 512))# 目前tile大小规定为512 多tile的方式需要再测试
    # tile_height    = int(data.get('tile_height', 512))# 目前tile大小规定为512 多tile的方式需要再测试
    tile_width  = 512
    tile_height = 512
    mask_blur      = float(data.get('mask_blur', 8.0))
    padding        = int(data.get('padding', 32))
    upscaler       = data.get('upscaler', None)# placeholder 用于以后的超分模型
    seams_fix      = data.get('seams_fix', {})
    seams_fix_enable= bool(seams_fix.get('enable', False))# 目前没有开启缝隙修复

    if 'alwayson_scripts' in data:
        if "controlnet" in data['alwayson_scripts']:
            if "args" in data['alwayson_scripts']['controlnet']:
                controlnet_args = data['alwayson_scripts']['controlnet']['args'][0]
                if "enabled" in data['alwayson_scripts']['controlnet']['args'][0]:
                    if data['alwayson_scripts']['controlnet']['args'][0]['enabled']==False:
                        controlnet_image= None
                        flag = False
                    else:
                        flag = True
                else:
                    flag = False
                    controlnet_image= None
                if len(data['alwayson_scripts']['controlnet']['args']) ==1  and flag:
                    args_info = data['alwayson_scripts']['controlnet']['args'][0]
                    if 'image' in args_info:
                        controlnet_image = data['alwayson_scripts']['controlnet']['args'][0]['image']
                        # base64 to image
                        controlnet_image = base64.b64decode(controlnet_image)
                        controlnet_image = Image.open(io.BytesIO(controlnet_image))
                else:
                    controlnet_image = None
    with app.app_context():
        pipeline = app.config['pipeline']  # 获取 pipeline 变量
        assert [width, height] in SHAPES
        pipeline.set_height_width(height, width)
        try:
            pipeline.scheduler = sampler_index
            img_pil = pipeline.wrap_upscale(
                prompt=prompt,
                negative_prompt=negative_prompt,
                init_image=init_image,
                mask=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_img = controlnet_image,
                seeds = [seed],
                subseeds = [subseed],
                subseed_strength=subseed_strength,
                seed_resize_from_h=seed_resize_from_h,
                seed_resize_from_w=seed_resize_from_w,
                controlnet_args = controlnet_args,
                upscale_factor = upscale_factor,
                target_width = target_width,
                target_height = target_height,
                upscale_type = upscale_type,
                mask_blur = mask_blur,
                tile_width = tile_width,
                tile_height = tile_height,
                padding   = padding,
                seams_fix_enable = seams_fix_enable,
                upscaler = upscaler,
                seams_fix = seams_fix,
            )
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(trace)
            print(e)
            print("error")
            raise e
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # 构建JSON响应
    response = jsonify({'images': [ret_img_b64]})
    # 设置响应头
    response.headers['Content-Type'] = 'application/json'
    return response


@app.route("/basicupscale", methods=['POST'])
def process_upscale_with_upscale_model():
    # =================================================#
    # 只接受两个输入：img和upsacle ratio
    # =================================================#
    # 从请求中获取 JSON 数据
    data = request.get_json()
    # 从 JSON 数据中获取所需数据
    init_image_b64 = data['init_images'][0]
    upscale_factor = None
    if data.get('upscale_factor') is None:
        if data.get('target_width') is None or data.get('target_height') is None:
            raise Exception("upscale_factor or target_width and target_height must be provided")
        else:
            upscale_factor = None
            target_width = int(data.get('target_width'))
            target_height = int(data.get('target_height'))
    else:
        upscale_factor = int(data.get('upscale_factor'))
    init_image = None
    if init_image_b64:
        try:
            init_image_bytes = BytesIO(base64.b64decode(init_image_b64))
            init_image = Image.open(init_image_bytes)
        except Exception as e:
            print("init image base64 decode error")
            print(init_image_b64)
            print("please check the image is/not is correct")
            print("please be sure that the image MUST be base64. Any other format will cause error")
            print("error:",e)
            raise e
    else:
        raise Exception("init image must be provided")
    if upscale_factor is None:
        source_width = init_image.size[0]
        source_height = init_image.size[1]
        assert target_width / source_width == target_height / source_height
        upscale_factor = max(target_width / source_width, target_height / source_height)
    upsacle = UpscaleModel(tile_size=(200,200),padding=4,upscale_rate=upscale_factor,model_size=(200,200))
    img_pil = upsacle.extract_and_enhance_tiles(init_image, upscale_ratio=upscale_factor)
    buffer = io.BytesIO()
    img_pil.save(buffer, format='JPEG')
    ret_img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # 构建JSON响应
    response = jsonify({'images': [ret_img_b64]})
    # 设置响应头
    response.headers['Content-Type'] = 'application/json'
    return response


if __name__ == "__main__":
    # engine setup
    app.run(debug=False, port=7019, host="0.0.0.0", threaded=False)