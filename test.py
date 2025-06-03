import pprint
import argparse
import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import datasets
from utils import train_util, log_util, anomaly_util
from config.defaults import _C as config, update_config
from models.wresnet1024_cattn_tsm import ASTNet as get_net1
from models.wresnet2048_multiscale_cattn_tsmplus_layer6 import ASTNet as get_net2

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# --cfg experiments/sha/sha_wresnet.yaml --model-file output/shanghai/sha_wresnet/shanghai.pth GPUS [3]
# --cfg experiments/ped2/ped2_wresnet.yaml --model-file output/ped2/ped2_wresnet/ped2.pth GPUS [3]
def parse_args():
    parser = argparse.ArgumentParser(description='Test Anomaly Detection')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        default='config/shanghaitech_wresnet.yaml', type=str)
    parser.add_argument('--model-file', help='model parameters',
                        default='pretrained/shanghaitech.pth', type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = \
        log_util.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False       # TODO ? False
    config.freeze()

    gpus = [(config.GPUS[0])]
    # model = models.get_net(config)
    # if config.DATASET.DATASET == "S01":
    model = get_net1(config, pretrained=False)
    # else:
    #     model = get_net2(config, pretrained=False)
    logger.info('Model: {}'.format(model.get_name()))
    model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])
    logger.info('Epoch: '.format(args.model_file))



    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    test_dataset = eval('datasets.get_test_data')(config)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    mat_loader = datasets.get_label(config)
    mat = mat_loader()

    psnr_list = inference(config, test_loader, model)
    assert len(psnr_list) == len(mat), f'Ground truth has {len(mat)} videos, BUT got {len(psnr_list)} detected videos!'

    auc, fpr, tpr, scores = anomaly_util.calculate_auc(config, psnr_list, mat)
    np.save('Anoo.npy', scores)
    logger.info(f'AUC: {auc * 100:.1f}%')


def inference(config, data_loader, model):
    loss_func_mse = nn.MSELoss(reduction='none')

    model.eval()
    psnr_list = []
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df  # number of frames to process
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print('[{}/{}]'.format(i+1, len(data_loader)))
            psnr_video = []

            # compute the output
            video, video_name = train_util.decode_input(input=data, train=False)
            video = [frame.to(device=config.GPUS[0]) for frame in video]
            for f in tqdm.tqdm(range(len(video) - fp)):
                inputs = video[f:f + fp]
                output = model(inputs)
                target = video[f + fp:f + fp + 1][0]

                # compute PSNR for each frame
                # https://github.com/cvlab-yonsei/MNAD/blob/d6d1e446e0ed80765b100d92e24f5ab472d27cc3/utils.py#L20
                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                psnr = anomaly_util.psnr_park(mse_imgs)
                psnr_video.append(psnr)

            psnr_list.append(psnr_video)
    return psnr_list


#
# import torch
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# def load_model(model_path):
#     """
#     加载训练好的模型
#     """
#     # 替换为你的模型定义
#     from models.wresnet1024_cattn_tsm import ASTNet  # 替换为实际的模型定义文件和类
#     model = ASTNet(config='config/ped2_wresnet.yaml')
#     model.load_state_dict(torch.load(model_path))
#     model.eval()  # 切换到评估模式
#     return model

# 图像预处理
# def preprocess_frame(frame, input_size):
#     """
#     预处理单帧图像
#     frame: 输入帧 (numpy array)
#     input_size: 模型输入尺寸 (height, width)
#     """
#     resized_frame = cv2.resize(frame, input_size)
#     if len(resized_frame.shape) == 3 and resized_frame.shape[2] == 3:
#         resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
#     normalized_frame = resized_frame / 255.0  # 归一化到 [0, 1]
#     tensor_frame = torch.tensor(normalized_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
#     return tensor_frame
#
# # 可视化结果
# def visualize_result(frame, result):
#     """
#     可视化模型输出结果
#     frame: 原始帧
#     result: 模型输出 (假设为热图)
#     """
#     normalized_result = (result - result.min()) / (result.max() - result.min() + 1e-8)
#     heatmap = cv2.applyColorMap((normalized_result * 255).astype(np.uint8), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
#
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title("Original Frame")
#     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.axis("off")
#
#     plt.subplot(1, 2, 2)
#     plt.title("Detection Result")
#     plt.imshow(overlay)
#     plt.axis("off")
#     plt.show()

#
# def vis():
#     model_path = "my_output_ssp_dyn_sf0926/R01/ped2_wresnet/epoch_200.pth"  # 替换为你的模型路径
#     frame_path = "/data1/yqz/IPAD_dataset/R01/testing/frames/01/085.jpg"  # 替换为你的测试帧路径
#     input_size = (256, 224)  # 替换为模型的输入尺寸
#
#     # 加载模型
#     model = get_net1(config, pretrained=False)
#
#     # 读取测试帧
#     frame = cv2.imread(frame_path)
#     if frame is None:
#         print(f"无法加载帧: {frame_path}")
#         return
#
#     # 预处理帧
#     input_tensor = preprocess_frame(frame, input_size)
#
#     # 推理
#     with torch.no_grad():
#         output = model(input_tensor)
#
#     # 假设输出是二维热图
#     output_heatmap = output.squeeze().numpy()
#
#     # 可视化
#     visualize_result(frame, output_heatmap)




if __name__ == '__main__':
    main()
