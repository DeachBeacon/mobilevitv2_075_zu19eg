from pathlib import Path
import math
import re
import html
import json
import zipfile
from PIL import Image, ImageDraw, ImageFont
import markdown

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FONT = 'C:/Windows/Fonts/msyh.ttc'
COLORS = {
    'pc': ('#F0EAFE', '#7C57C2'),
    'ps': ('#E9F3FF', '#3178B9'),
    'dpu': ('#FFF2D7', '#BA821F'),
    'ok': ('#E8F5EE', '#34815C'),
    'warn': ('#FFF0EB', '#B66449'),
    'plain': ('#F2F5F9', '#77879D'),
}

class Diagram:
    def __init__(self, stem, title, subtitle, width, height):
        self.stem, self.w, self.h = stem, width, height
        self.im = Image.new('RGB', (width, height), '#FFFFFF')
        self.draw = ImageDraw.Draw(self.im)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<rect width="100%" height="100%" fill="white"/>']
        self.text(52, 38, title, 34, '#172D47')
        self.text(54, 92, subtitle, 20, '#607185')
        self.draw.line((52, 134, width-52, 134), fill='#D9E1EB', width=2)
        self.svg.append(f'<path d="M52 134H{width-52}" stroke="#D9E1EB" stroke-width="2"/>')
        self.nodes = {}

    def text(self, x, y, text, size=22, fill='#20354E', anchor='left'):
        font = ImageFont.truetype(FONT, size)
        if anchor == 'center':
            length = self.draw.textlength(text, font=font)
            dx = x-length/2
        else:
            dx = x
        self.draw.text((dx, y), text, font=font, fill=fill)
        self.svg.append(f'<text x="{x}" y="{y+size}" font-family="Microsoft YaHei, sans-serif" font-size="{size}" fill="{fill}" text-anchor="{"middle" if anchor=="center" else "start"}">{html.escape(text)}</text>')

    def box(self, key, x, y, w, h, title, lines=(), kind='plain', title_size=23):
        for item,size in [(title,title_size)]+[(line,20) for line in lines]:
            assert self.draw.textlength(item,font=ImageFont.truetype(FONT,size)) < w-24,(self.stem,key,item)
        bg, border = COLORS[kind]
        self.draw.rounded_rectangle((x,y,x+w,y+h),radius=14,fill=bg,outline=border,width=2)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{bg}" stroke="{border}" stroke-width="2"/>')
        self.text(x+w/2, y+16, title, title_size, '#19364F', 'center')
        if lines:
            line_height = 29
            total = len(lines)*line_height
            first = max(y+55, y+(h-total)/2+10)
            for i,line in enumerate(lines):
                self.text(x+w/2, first+i*line_height, line, 20, '#3D5369', 'center')
        self.nodes[key] = (x,y,w,h)

    def points(self, pts, label=None, label_at=None, color='#75889E'):
        self.draw.line(pts, fill=color, width=3, joint='curve')
        points=' '.join(f'{x},{y}' for x,y in pts)
        self.svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
        a,b=pts[-2],pts[-1]
        ang=math.atan2(b[1]-a[1],b[0]-a[0])
        l=12
        tip=[b,(b[0]-l*math.cos(ang-.48),b[1]-l*math.sin(ang-.48)),(b[0]-l*math.cos(ang+.48),b[1]-l*math.sin(ang+.48))]
        self.draw.polygon(tip,fill=color)
        self.svg.append('<polygon points="'+' '.join(f'{x:.1f},{y:.1f}' for x,y in tip)+f'" fill="{color}"/>')
        if label and label_at:
            self.text(*label_at,label,18,'#607185','center')

    def arrow(self, src, dst, start='bottom', end='top'):
        def p(key,side):
            x,y,w,h=self.nodes[key]
            return {'top':(x+w/2,y),'bottom':(x+w/2,y+h),'left':(x,y+h/2),'right':(x+w,y+h/2)}[side]
        self.points([p(src,start),p(dst,end)])

    def foot(self, text):
        self.text(54,self.h-43,text,18,'#607185')

    def save(self):
        self.im.save(FIG/(self.stem+'.png'))
        (FIG/(self.stem+'.svg')).write_text('\n'.join(self.svg+['</svg>']),encoding='utf-8')

def vertical(stem,title,subtitle,rows,foot='',width=1280):
    height=195+len(rows)*145+30
    d=Diagram(stem,title,subtitle,width,height)
    for i,(heading,lines,kind) in enumerate(rows):
        d.box(str(i),90,165+i*145,width-180,110,heading,lines,kind)
        if i:d.arrow(str(i-1),str(i))
    d.foot(foot)
    d.save()
    return d

vertical('图1_设计流程','图1  设计与部署流程','模型切分、数值校验、量化构建与板端反馈',[
    ('分析源模型与兼容性',['识别卷积、注意力、归一化和重排；确定PS/PL边界'],'pc'),
    ('PC：切分六段并处理参数',['BN折叠 / CLE → 验证浮点结果 → 应用已确定的前端补偿'],'pc'),
    ('PC：导出模型与接口清单',['六段PB、JSON/NPZ；parts.json / inputs.json'],'pc'),
    ('Vitis AI Docker：校准、量化及原生评估',['图像 / 中间特征 → 三段INT8；完整混合链路评价'],'dpu'),
    ('Vitis AI / Runtime Docker：编译与链接',['量化PB → ELF → AArch64 SO → 完整部署包'],'dpu'),
    ('开发板：推理、统计与迭代',['单图 / 批量测试；精度、分段耗时、输入范围 → 反馈优化'],'ok'),
], '等价性验证针对切分及BN/CLE；偏置补偿效果另行评估。')

d=Diagram('图2_软件总体框图','图2  软件总体框图','离线构建工具 + 板端PS/PL协同推理',1600,1130)
d.box('pc',55,175,450,160,'PC：模型准备',['训练权重 → H5 → 冻结PB','六段切分 / BN / CLE / 模型导出'],'pc')
d.box('vai',575,175,450,160,'Vitis AI Docker',['中间特征校准 / INT8量化','原生FixNeuron评估 / ELF编译'],'dpu')
d.box('link',1095,175,450,160,'Runtime Docker',['AArch64共享库链接','SO / 配置 / PS模型 / Python程序'],'pc')
d.arrow('pc','vai','right','left');d.arrow('vai','link','right','left')
d.text(800,385,'完整部署包拷贝至AXU19EG开发板',24,'#42627E','center')
d.text(55,445,'板端执行通路（箭头为单张图片的先后顺序）',22,'#607185')
cards=[('input',55,505,'PS：图像预处理',['256×256 RGB图片','FP32 [1,256,256,3]'],'ps'),
       ('front',445,505,'DPU：front',['前端卷积','FP32输出 [1,32,32,96]'],'dpu'),
       ('ps1',835,505,'PS：ps1',['Transformer所在计算段1','输出 [1,32,32,192]'],'ps'),
       ('mv4',1225,505,'DPU：mv2_4',['MV2卷积与下采样','输出 [1,16,16,288]'],'dpu'),
       ('ps2',1225,755,'PS：ps2',['Transformer所在计算段2','输出 [1,16,16,288]'],'ps'),
       ('mv5',835,755,'DPU：mv2_5',['MV2卷积与下采样','输出 [1,8,8,384]'],'dpu'),
       ('ps3',445,755,'PS：ps3',['Transformer及分类后端','输出 [1,10] logits'],'ps'),
       ('result',55,755,'PS：结果输出',['argmax → 类别0～9','CSV / 精度 / 耗时 / 诊断'],'ok')]
for k,x,y,t,ls,c in cards:d.box(k,x,y,320,165,t,ls,c,22)
for a,b in [('input','front'),('front','ps1'),('ps1','mv4')]:d.arrow(a,b,'right','left')
d.arrow('mv4','ps2')
for a,b in [('ps2','mv5'),('mv5','ps3'),('ps3','result')]:d.arrow(a,b,'left','right')
d.text(800,990,'PS段：Python / NumPy FP32        DPU段：内部INT8，N2Cube以FP32接口传输',22,'#42627E','center')
d.foot('三个DPU段对应三个依次调用的kernel；不表示三个物理DPU实例。')
d.save()

d=Diagram('图3_PC模型处理','图3  PC端模型处理函数流程','入口函数 → 图切分 → 参数处理 → 导出与接口检查',1400,1130)
d.box('src',170,170,1060,95,'源模型转换',['PT → H5 → source.pb；固定输入 [1,256,256,3]'],'pc')
d.box('prep',170,310,1060,100,'prepare_pc() → prepare()',['环境 / 源图 / 参数身份检查；准备图像缓存'],'pc')
d.arrow('src','prep')
d.box('split',170,455,1060,95,'split_graph(original)',['前端祖先子图 + 后段切点占位符；检查DPU前端算子'],'pc')
d.arrow('prep','split')
d.box('front',60,610,610,175,'前端：fold_front() → apply_scales()',['feature_groups()约束残差尺度','采用固定CLE尺度；再add_output_bias()'],'pc',22)
d.box('tail',730,610,610,175,'后段：cut()循环生成五个子图',['PS1 / MV2_4 / PS2 / MV2_5 / PS3','MV2：fold_front → equalize → apply_scales'],'pc',22)
d.points([(700,550),(700,580),(365,580),(365,610)])
d.points([(700,580),(1035,580),(1035,610)])
d.box('export',170,850,1060,110,'prepare_numpy_export() → check_prepared()',['拓扑排序 / 算子白名单 / 数值权重 → 六段PB、JSON/NPZ与哈希清单'],'ok',23)
d.points([(365,785),(365,815),(700,815),(700,850)])
d.points([(1035,785),(1035,815),(700,815)])
d.foot('各段以固定FP32张量连接；切分检查唯一输入，防止遗漏跨段依赖。')
d.save()

vertical('图4_量化编译组装','图4  量化、编译与组装函数流程','PC模型不在Docker中重新导出',[
    ('check_prepared() → init_build()',['校验源图 / 参数 / PC产物；绑定构建目录'],'pc'),
    ('cache_mv2() → Vitis AI量化',['父量化前缀 + PS1/PS2 → MV2特征缓存；分别量化三个DPU段'],'dpu'),
    ('postprocess()',['lock_input：同步输入位置；correct_mv4：更新MV2_4偏置'],'dpu'),
    ('quantized() / evaluate()',['固化量化产物身份；可选执行完整链路原生精度评估'],'ok'),
    ('Vitis AI编译 → compiled()',['三段deploy PB → ELF；检查单DPU且无CPU回退；提取节点名'],'dpu'),
    ('assemble(link=True)',['交叉链接SO → 核对DPU节 → 复制PS模型 / 程序 / 配置'],'pc'),
    ('部署包检查与输出',['inspect_libraries()；记录BUILD_INFO.json'],'ok'),
], '前端scale=32；两个MV2 scale=1；编译与运行配置必须对应。')

d=Diagram('图5_入口与批量任务','图5  板端入口、初始化与批量任务流程','单图与批量入口共用HybridRunner；DPU任务只初始化一次',1500,1320)
d.box('single',65,175,650,120,'单图main()',['图片或NPY → 库路径检查 → 加载输入'],'ps')
d.box('batch',785,175,650,120,'批量main() → run_batch(args)',['扫描图片 / 标签校验 / 新输出目录'],'ps')
d.box('init',280,365,940,160,'HybridRunner.__init__()',['配置与SO哈希 → inspect_libraries() → 相邻形状校验','加载PS模型 → dpuOpen → DpuStage初始化与尺度校验'],'pc')
d.points([(390,295),(390,330),(750,330),(750,365)])
d.points([(1110,295),(1110,330),(750,330)])
d.box('load',280,585,940,105,'load_image() → preprocess_rgb()',['解码 / BGR→RGB / 固定尺寸检查 / mean-std标准化'],'ps')
d.arrow('init','load')
d.box('warm',280,750,940,105,'批量：首张有效图预热；随后逐张调用predict()',['模型与任务复用；预热耗时单独记录'],'ps')
d.arrow('load','warm')
d.box('record',280,915,940,110,'记录并汇总结果',['predictions.csv；summary.json；精度 / 混淆矩阵 / 各段耗时'],'ok')
d.arrow('warm','record')
d.points([(1220,970),(1360,970),(1360,637),(1220,637)],'下一张',(1350,785))
d.box('close',280,1090,940,110,'finally：保存诊断并close()',['save_input_report() → 反向释放task/kernel → dpuClose'],'plain')
d.arrow('record','close')
d.foot('单图入口执行一次predict后输出JSON；循环、预热与批量统计属于run_batch。')
d.save()

vertical('图6_六段混合调度','图6  HybridRunner.predict()六段调度','所有段间数据均为FP32；每段分别记录墙钟耗时',[
    ('输入 image',['float32 [1,256,256,3]；已完成统一预处理'],'ps'),
    ('self.front.predict(image)',['DPU front → [1,32,32,96]'],'dpu'),
    ('ps1.predict(value)',['PS Transformer所在计算段1 → [1,32,32,192]'],'ps'),
    ('mv2_4.predict(value)',['DPU中间MV2 → [1,16,16,288]'],'dpu'),
    ('ps2.predict(value)',['PS Transformer所在计算段2 → [1,16,16,288]'],'ps'),
    ('mv2_5.predict(value)',['DPU中间MV2 → [1,8,8,384]'],'dpu'),
    ('ps3.predict(value)',['PS Transformer与分类后端 → [1,10]'],'ps'),
    ('输出检查与类别选择',['有限性 / 10维检查 → argmax；返回class_index、logits、seconds'],'ok'),
], 'DPU耗时包含数据传输与量化/反量化；Python调度按序执行。')

d=Diagram('图7_DPU执行与诊断','图7  DpuStage.predict()执行流程','初始化已校验边界和实际输入尺度s',1500,1510)
d.box('in',300,175,900,110,'校验与连续化输入value',['浮点类型 / 精确形状 / 有限值 → 连续float32数组'],'ps')
d.box('range',300,340,900,130,'统计min / max与越界元素',['合法端点区间 [-128/s, 127/s]','记录段名、调用数、最值、越界比例'],'plain')
d.arrow('in','range')
d.box('error',45,555,440,145,'越界且策略为error',['保存首次越界输入','抛出异常，停止当前推理'],'warn')
d.box('valid',530,555,440,145,'范围正常',['直接使用输入','保持运行库原有舍入方式'],'ok')
d.box('clip',1015,555,440,145,'越界且策略为clip',['按该段实际尺度饱和裁剪','更新裁剪计数'],'warn')
d.points([(750,470),(750,510),(265,510),(265,555)])
d.points([(750,510),(750,555)])
d.points([(750,510),(1235,510),(1235,555)])
d.box('set',300,790,900,100,'dpuSetInputTensorInHWCFP32()',['写入FP32特征；check_status()检查返回码'],'dpu',22)
d.points([(750,700),(750,790)])
d.points([(1235,700),(1235,745),(750,745)])
d.box('run',300,945,900,100,'dpuRunTask()',['执行当前DPU kernel；check_status()'],'dpu')
d.arrow('set','run')
d.box('get',300,1100,900,100,'dpuGetOutputTensorInHWCFP32()',['取回已反量化FP32输出'],'dpu',22)
d.arrow('run','get')
d.box('out',300,1255,900,110,'检查输出 → reshape → copy()',['验证元素数与有限性；复制后交给下一PS段'],'ok')
d.arrow('get','out')
d.foot('输入范围报告包括预热与被拒绝的输入；默认策略为error。')
d.save()

d=Diagram('图8_NumPy执行器','图8  FastTail与NumPy底层算子流程','同一JSON/NPZ支持参考实现与快速实现',1500,1390)
d.box('init',200,175,1100,115,'FastTail.__init__()',['读JSON / NPZ → _optimize() → 建立张量引用计数'],'pc')
d.box('opt',200,345,1100,130,'_optimize(nodes)',['_const_of追踪参数；_add_bias合并偏置','BN / BiasAdd融合 → 消除直通节点 → 重定向边和输出端口'],'pc')
d.arrow('init','opt')
d.box('predict',200,535,1100,115,'predict(features)',['检查输入形状 → 按拓扑顺序取节点输入'],'ps')
d.arrow('opt','predict')
d.box('conv',55,725,650,160,'卷积节点：_conv()',['1×1：直接矩阵乘法','一般卷积：连续im2col + matmul','深度卷积：按卷积核位置累加'],'ps')
d.box('eval',795,725,650,160,'其他节点：_evaluate()',['Transpose / Reshape / SplitV','归约 / 归一化基础运算 / ReLU','Softmax / MatMul / 加减乘等'],'ps')
d.points([(750,650),(750,685),(380,685),(380,725)])
d.points([(750,685),(1120,685),(1120,725)])
d.box('release',200,955,1100,115,'保存必要输出并释放不再使用的中间张量',['按端口保存输出；引用计数减一，计数归零时释放'],'ps')
d.points([(380,885),(380,920),(750,920),(750,955)])
d.points([(1120,885),(1120,920),(750,920)])
d.points([(1300,1012),(1465,1012),(1465,592),(1300,592)],'下一节点',(1390,665))
d.box('out',200,1135,1100,110,'检查该段输出形状并返回',['PS1 / PS2返回中间特征；PS3返回10类logits'],'ok')
d.arrow('release','out')
d.foot('融合在加载时完成；推理阶段不重复执行可合并的BN和偏置节点。')
d.save()

d=Diagram('图9_精度优化闭环','图9  采集、补偿与完整链路评估','同输入对齐 → 参数补偿 → 重建与验证',1500,1370)
d.box('capture',220,175,1060,130,'capture_features.main()：板端对照采集',['保存实际MV2输入、logits及部分诊断输出','manifest绑定图片、NPZ及三个SO哈希'],'ps')
d.box('fit',220,365,1060,140,'fit()：同输入浮点 / 真实FixNeuron对照',['MeanError.add() → MeanError.value()','delta[c] = mean(reference[...,c] - quantized[...,c])'],'pc')
d.arrow('capture','fit')
d.box('front',55,585,650,160,'前端补偿',['add_output_bias()在PC浮点图应用偏移','再执行量化；外部输出形状不变'],'pc')
d.box('mv4',795,585,650,160,'MV2_4补偿',['原始末层偏置 + delta → 原FixNeuron网格','correct_mv4()同步eval与deploy偏置'],'dpu')
d.points([(750,505),(750,545),(380,545),(380,585)])
d.points([(750,545),(1120,545),(1120,585)])
d.box('evaluate',220,825,1060,125,'evaluate()：完整混合链路评价',['前端量化 → PS1 → MV2_4量化 → PS2 → MV2_5量化 → PS3','统计类别、准确率和MV2输入越界'],'ok',22)
d.points([(380,745),(380,785),(750,785),(750,825)])
d.points([(1120,745),(1120,785),(750,785)])
d.box('accept',220,1015,1060,130,'编译 / 链接 → 最佳组合上板验证',['检查同一版本的精度、耗时与范围报告','另用独立图片集评价泛化表现'],'ok')
d.arrow('evaluate','accept')
d.foot('历史最终候选：Docker原生258/300（86%）；最佳组合板端表现仍需实测。')
d.save()

# Portable browser / print version. All figures are embedded so the HTML is one file.
md=(ROOT/'报告正文.md').read_text(encoding='utf-8')
body=markdown.markdown(md,extensions=['tables','fenced_code','sane_lists'])
import base64
for path in FIG.glob('*.png'):
    body=body.replace('figures/'+path.name,'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode())
style='''
body{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;background:#eef2f7;color:#20334c;line-height:1.85;margin:0;font-size:16px}
main{max-width:1120px;margin:36px auto;padding:42px 58px;background:white;box-shadow:0 8px 40px #10233c12}
h1{font-size:30px;line-height:1.5;color:#173955;margin:0 0 30px}h2{font-size:25px;border-bottom:2px solid #abc1d3;padding-bottom:10px;margin-top:45px}
h3{font-size:22px;margin-top:40px;color:#234c6b}h4{font-size:19px;margin-top:34px;color:#285674}p{margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;line-height:1.65;margin:24px 0;table-layout:auto}th,td{border:1px solid #cfdae5;padding:10px;vertical-align:top;overflow-wrap:anywhere}th{background:#eaf1f7;color:#1d4969;text-align:left}tr:nth-child(even){background:#f8fafc}
code{font-family:Consolas,"Microsoft YaHei",monospace;font-size:.91em;color:#345c80;background:#f0f4f8;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}img{display:block;max-width:100%;height:auto;margin:25px auto;border:1px solid #e0e7ee;border-radius:8px}li{margin:9px 0}
.note{font-size:14px;background:#edf4fa;border-left:4px solid #4e82a9;padding:12px 18px;color:#526c81;margin-bottom:30px}
@media print{body{background:white;font-size:10.5pt}main{margin:0;padding:0;box-shadow:none;max-width:none}h1{font-size:20pt}h2{font-size:16pt}h3{font-size:14pt}h4{font-size:12pt}table{font-size:8.5pt}img{max-height:240mm;max-width:100%;object-fit:contain;break-inside:avoid}h2,h3,h4{break-after:avoid}tr{break-inside:avoid}.note{display:none}@page{size:A4;margin:17mm}}
'''
html_text='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MobileViTv2 PS/PL协同部署软件系统报告</title><style>'+style+'</style><main><div class="note">依据三段项目对话与当前源代码整理。图示为软件设计图；历史原生评估与板端实测分别标注。此文件已内嵌全部图片，可单独保存或打印。</div>'+body+'</main></html>'
(ROOT/'报告正文.html').write_text(html_text,encoding='utf-8')

# Check constrained sections conservatively: count all non-whitespace characters,
# including punctuation, letters, digits and the numbered innovation items.
counts={}
for start,end,label,limit in [('## 1.1','## 1.2','1.1',400),('## 1.2','## 1.3','1.2',200),('## 1.3','![图1','1.3',200)]:
    text=md.split(start,1)[1].split(end,1)[0].split('\n',1)[1]
    n=len(re.sub(r'\s+','',text))
    counts[label]={'non_whitespace_characters':n,'limit':limit,'passed':n<=limit}
    assert n<=limit,(label,n)
for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',md):
    assert (ROOT/link).is_file(),link

# A contact sheet for visual quality inspection.
thumb_w=450
tiles=[]
for p in sorted(FIG.glob('*.png')):
    im=Image.open(p)
    im.thumbnail((thumb_w,620))
    tile=Image.new('RGB',(490,690),'#eaf0f6')
    tile.paste(im,((490-im.width)//2,25))
    ImageDraw.Draw(tile).text((18,655),p.stem,font=ImageFont.truetype(FONT,19),fill='#20334c')
    tiles.append(tile)
sheet=Image.new('RGB',(1470,2070),'white')
for i,tile in enumerate(tiles):sheet.paste(tile,((i%3)*490,(i//3)*690))
sheet.save(ROOT/'配图检查总览.png')

(ROOT/'字数与文件检查.json').write_text(json.dumps({'sections':counts,'figures':9,'all_image_links_exist':True},ensure_ascii=False,indent=2),encoding='utf-8')
archive_path=ROOT.parent/'软件系统报告与配图_20261103.zip'
deliverables=[ROOT/name for name in ['README.md','报告正文.md','报告正文.html','核对依据.md','字数与文件检查.json']]
deliverables+=sorted(FIG.glob('*.png'))+sorted(FIG.glob('*.svg'))
with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in deliverables:
        archive.write(path,(Path(ROOT.name)/path.relative_to(ROOT)).as_posix())
with zipfile.ZipFile(archive_path) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist())==23
print(json.dumps({'counts':counts,'figures':9,'html_bytes':(ROOT/'报告正文.html').stat().st_size,
                  'archive':str(archive_path),'archive_files':len(deliverables)},ensure_ascii=False))
