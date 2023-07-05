wget https://github.com/sophgo/tpu-perf/releases/download/v1.2.17/tpu_perf-1.2.17-py3-none-manylinux2014_aarch64.whl 
pip3 install tpu_perf-1.2.17-py3-none-manylinux2014_aarch64.whl && rm -rf tpu_perf-1.2.17-py3-none-manylinux2014_aarch64.whl
pip3 install -r requirements.txt -i https://pypi.douban.com/simple
mkdir models
mkdir models/basic
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/iCUZMB1NF && unzip controlnet.zip && rm -rf controlnet.zip && mv controlnet models/
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/SzHkJPo06 && unzip deliberate-lora_pixarStyleLora_lora128-unet-2 && rm -rf deliberate-lora_pixarStyleLora_lora128-unet-2.zip && mv deliberate-lora_pixarStyleLora_lora128-unet-2 models/basic/
ptyhon3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/1Z4DdzGsk && unzip other.zip && rm -rf other.zip && mv other models/

