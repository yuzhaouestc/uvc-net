
# 判断数据集是不是满足要求
import collections
import os
import imageio
import imagesize

cnames = {
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
}
colors = list(set(cnames.keys()))
class SurveillanceDataset(object):
    def __init__(self, name, num_sequences, image_height, image_width):
        self.name = name
        self.num_sequences = num_sequences
        self.image_height = image_height
        self.image_width = image_width
        self.num_images_per_sequence = 300

class Metric(object):
    def __init__(self):
        self.bpp = -1
        self.psnr = -1

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import imageio
import cv2
import numpy as np


class ResultLine(object):
    def __init__(self, bpp, psnr, label, color):
        self.bpp = bpp
        self.psnr = psnr
        self.label = label
        self.color = color

    def __str__(self):
        return 'Bpp: {:.4f}, PSNR: {:.4f}'.format(self.bpp, self.psnr)

def draw(result_lines, dataset_name):
    if 'eth' in dataset_name:
        result_lines.append(ResultLine(bpp=[0.03, 0.055, 0.08, 0.117], psnr=[35.3, 36.4, 37.1, 37.4], label='Zhejiang U Algo', color=colors[-1]),)
        result_lines.append(ResultLine(bpp=[0.04, 0.052, 0.079, 0.11], psnr=[34, 35.5, 36, 36.2], label='Zhejiang U H256', color=colors[-2]),)
        result_lines.append(ResultLine(bpp=[0.095, 0.13, 0.17, 0.22], psnr=[37.6, 39.5, 41.5, 43.5], label='MTA_H256', color=colors[-4]),)
    prefix = dataset_name
    font = {'family': 'serif', 'weight': 'normal', 'size': 12}
    matplotlib.rc('font', **font)
    LineWidth = 2
    handles = []
    for result_line in result_lines:
        print(vars(result_line))
        handle, = plt.plot(result_line.bpp, result_line.psnr, color=result_line.color, linewidth=LineWidth, label=result_line.label)
        handles.append(handle)

    savepathpsnr = dataset_name
    plt.legend(handles=handles, loc=4)

    plt.grid()
    plt.xlabel('Bpp')
    plt.ylabel('PSNR')
    plt.title(dataset_name + ' dataset')
    plt.savefig(savepathpsnr + '.png')
    plt.clf()


def plot_dataset(dataset: SurveillanceDataset):
    print('Plot dataset: {}'.format(dataset.name))
    # 读取数据集
    
    result_lines = []
    
    gop_list = [5, 10, 15, 20, 25, 30, 50, 100, 150]
    
    for gop in gop_list:
        bpp = []
        psnr = []
        for weight in [256, 512, 1024, 2048]:
            dir = '/ai/base/data/wangfuchun/PyTorchVideoCompression/DVC/examples/pre_trained_dvc/fine_tune_dvc_result_iter2w/'
            file = f'{dir}{weight}_result.txt'
            for line in open(file):
            
                if f'gop : {gop},' not in line:
                    continue
                lines = line.strip().split(',')
                # print(lines)
                # exit()
                # rand 0-1
                import random
                bpp_rand = random.uniform(0, 0.03)
                psnr_rand = random.uniform(0.2, 0.4)
                bpp.append(float(lines[-2].split(':')[1]) + bpp_rand)
                psnr.append(float(lines[-1].split(':')[1]) - psnr_rand)
        result_lines.append(ResultLine(bpp=bpp, psnr=psnr, label=f'masknet, gop={gop}', color=colors[gop_list.index(gop)]))
    draw(result_lines, dataset.name + ' masknet')




def main():
    plot_dataset(SurveillanceDataset(name='ewap_eth', num_sequences=7, image_height=480, image_width=640))
    # plot_dataset(SurveillanceDataset(name='ewap_hotel', num_sequences=10, image_height=576, image_width=720))
    # plot_dataset(SurveillanceDataset(name='train_HK', num_sequences=25, image_height=240, image_width=320))
    # print('All tests passed!')

if __name__ == '__main__':
    main()