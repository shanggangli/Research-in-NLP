import torch
from transformers import BertTokenizer,BertModel

text = ["冠脉造影+pci记录患者于16:40入导管室，平卧，常规消毒铺巾，局麻下以seldinger法穿刺rra，置6fr动脉鞘。用5ftig管行左右冠状动脉造影示:"
        "lm光滑规则、未见明显狭窄，lad近段90%狭窄，中远段弥漫型狭窄，最窄处90%，lcx中段50%狭窄，om1近中段80%狭窄，rca中段40%狭窄，"
        "pda近段长病变，约90%狭窄，冠脉右优型。经与患者、家属沟通后，决定行pci治疗。置入6fjr4.0至右冠口，送runthrough导丝通过病变至pda远端，"
        "用sipphire1.5*15mm球囊以12atm*5秒预扩病变处，后于rpda近段植入endeavorresolute2.25*30mm药物支架一枚,以12atm*4秒释放，"
        "造影示:支架贴壁良好，无残余狭窄，timi3级血流。换6frebu3.5指引导管至lm开口，runthrough导丝顺利通过lad病变并送至lad远端，"
        "用sapphire1.5*15mm球囊以14atm*5秒预扩lad中远段病变处，后于lad中远段病变处植入endeavorresolute2.5*30mm药物支架一枚，"
        "以12atm*4秒释放，用voyagernc2.75*12mm以最高压18atm*5秒后扩，造影示:支架贴壁良好，无残余狭窄，"
        "timi3级血流；于lad近段病变处植入endeavorresolute3.0*18mm药物支架一枚，以12atm*4秒释放，"
        "用voyagernc2.75*12mm以最高压20atm*5秒后扩，造影示:支架贴壁良好，无残余狭窄，timi3级血流。"
        "lcx、om1病变可23个月后再行介入治疗。术毕拔管，压迫止血。术中共用肝素7000u，造影剂200毫升，术中无不适，术后安全送返病房。"]

tokenize = BertTokenizer.from_pretrained("bert-base-chinese")

tokens = tokenize.tokenize(str(text))
print("中文分词来一个：",tokens)

token = BertTokenizer.from_pretrained("bert-base-chinese")
input_ids = token(tokens)
print(input_ids)
word = token.convert_ids_to_tokens(input_ids)
print(word)