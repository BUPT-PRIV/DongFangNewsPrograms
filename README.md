# DongFangNewsPrograms
NLP 和 CV 编目新闻节目。
本新闻项目需要下载以下几个资源并放在相应目录下。
1.预训练模型下载地址：
bert_Chinese：模型https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz
解压后文件放入：
--Bert-3-class/
  --bert_pretrain/
    --bert-base-chinese.tar.gz.2
    --pytorch_model.bin
--Bert-10-class/
  --bert_pretrain/
    --bert-base-chinese.tar.gz.2
    --pytorch_model.bin
--Bert-3000-class/
  --bert_pretrain/
    --bert-base-chinese.tar.gz.2
    --pytorch_model.bin
2.GPT2总结模型
链接：https : //pan.baidu.com/s/1atsbABI7Lq5HQNctC11E5g
提取码：grtn
提取后放入：
--GPT2-Summary/
  --summary_model/
    --pytorch_model.bin
3.其他模型下载：
链接：https://pan.baidu.com/s/1lNsGycvPa_LRfZefWa3JDw 
提取码：2021 
提取后将：
10_bert.ckpt放入
--Bert-10-class/
  --newsdataset/
    --saved_dict/
      --bert.ckpt
      
3_bert.ckpt放入
--Bert-3-class/
  --newsdataset/
    --saved_dict/
      --bert.ckpt
      
3000_bert.ckpt放入
--Bert-3000-class/
  --newsdataset/
    --saved_dict/
      --bert.ckpt
      
pytorch_model.bin放入
--berts_pytorch_zh/
  --model/
    --bert-base-chinese/
      --pytorch_model.bin

运行：首先准备好视频文件，在终端输入：
python3 predict.py --path (视频文件绝对路径)
最终生成文件为result.json
