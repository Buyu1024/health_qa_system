from typing import TYPE_CHECKING

'''
paddleocr：解析图片中的文字，也可以进行表格识别
rapidocr_paddle 和 rapidocr_onnxruntime 两种导入方式
主要区别在于它们所使用的推理引擎和硬件支持
选择哪种方式最合适取决于你的硬件环境和性能需求。
当你有 GPU 且追求速度时：使用 rapidocr_paddle。PaddlePaddle 原生支持在 GPU 上推理 PaddleOCR 模型，速度更快。
当只有 CPU 且需要高效推理时：使用 rapidocr_onnxruntime。它在 CPU 上进行了优化，资源占用较低.
'''

def get_ocr(use_cuda: bool = False) -> "RapidOCR":
    # 优先使用 ONNX Runtime 版本,在 CPU 环境下更稳定
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    except ImportError:
        # 如果 ONNX Runtime 不可用,再尝试 PaddlePaddle 版本
        try:
            from rapidocr_paddle import RapidOCR
            '''
            det_use_cuda=True：启用检测模型的GPU加速。cls_use_cuda=True：启用分类模型的GPU加速。rec_use_cuda=True：启用识别模型的GPU加速。
            '''
            ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
        except ImportError:
            raise ImportError("请安装 rapidocr_onnxruntime 或 rapidocr_paddle")
    return ocr
