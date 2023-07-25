# wget https://github.com/sophgo/tpu-perf/releases/download/v1.2.17/tpu_perf-1.2.17-py3-none-manylinux2014_aarch64.whl 
python3 -m dfn --url http://219.142.246.77:65000/sharing/LiM00jkBJ && pip3 install sophon_arm-0.0.0-py3-none-any.whl && rm -rf ./sophon_arm-0.0.0-py3-none-any.whl
 
pip3 install tpu_perf-1.2.17-py3-none-manylinux2014_aarch64.whl
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if [ ! -d "./models" ]; then
    mkdir models
fi
if [ ! -d "./models/basic" ]; then
    mkdir models/basic
fi
if [ ! -d "./models/controlnet" ]; then
    mkdir models/controlnet
fi
if [ ! -d "./models/basic/babes20" ]; then
    python3 -m dfn --url http://219.142.246.77:65000/sharing/CQVgwv1Y5 && unzip babes20.zip && rm -rf babes20.zip && mv babes20 models/basic/
    rm -rf models/basic/babes20/text_encoder_1684x_f32.bmodel
    python3 -m dfn --url http://219.142.246.77:65000/sharing/Rpi4awF9I && mv encoder_1684x_f32.bmodel models/basic/babes20/text_encoder_1684x_f32.bmodel
fi

python3 -m dfn --url http://219.142.246.77:65000/sharing/63dydmQ6q && mv canny_multize.bmodel models/controlnet/
python3 -m dfn --url http://219.142.246.77:65000/sharing/86Rm1E7cl && mv tile_multize.bmodel models/controlnet/
